"""SGLang-style fixed dual-pool baseline.

Mirrors the static dual-pool partition that SGLang uses for hybrid
Mamba plus attention models. Two physically isolated regions are
pre-allocated at construction time: one ``PageTable`` for paged KV
pages, one ``BlockTable`` for fixed-size SSM state blocks. The
``mamba_ratio`` constructor argument selects the fraction of
``total_bytes`` assigned to the SSM pool.

Cross-pool eviction is intentionally absent, faithful to SGLang's
``MambaPool.alloc`` which simply returns ``None`` on exhaustion (no
fallback to the KV pool) at commit
``22012ba1bc2166f2280be2ad648ba732a0ff382b``:
https://github.com/sgl-project/sglang/blob/22012ba1bc2166f2280be2ad648ba732a0ff382b/python/sglang/srt/mem_cache/memory_pool.py

The SGLang production default is ``mamba_full_memory_ratio = 0.9``
declared at ``python/sglang/srt/server_args.py`` line 598 of the same
commit. This baseline ships with ``mamba_ratio = 0.5`` as a neutral
starting point for synthetic comparisons; deployments that want to
mirror SGLang's production default should pass ``mamba_ratio=0.9``.

The static partition cannot rebalance when one pool fills before the
other. Each ``allocate`` call whose target pool throws
:class:`CapacityError` adds the *other* pool's free bytes to the
running :attr:`pool_underused_bytes_<kv|ssm>` counter, surfacing the
amount of memory stranded by the rigidity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from cachepawl.allocator.base import Allocator, AllocatorStats
from cachepawl.allocator.baselines.common import (
    AllocatorContext,
    BackingStore,
    BlockTable,
    CapacityError,
    LRURequestTracker,
    PageHandle,
    PageTable,
    align_up,
)
from cachepawl.allocator.policy import EvictionPolicy
from cachepawl.models.spec import HybridModelSpec, LayerKind
from cachepawl.quant.dtypes import bytes_per_element

_KV_POOL_ID: int = 0
_SSM_POOL_ID: int = 1


class FixedDualPool(Allocator, AllocatorContext):
    """SGLang-style fixed dual-pool with cross-pool isolation."""

    def __init__(
        self,
        model_spec: HybridModelSpec,
        total_bytes: int,
        device: torch.device,
        mamba_ratio: float = 0.5,
        attention_page_tokens: int = 16,
        eviction: EvictionPolicy = EvictionPolicy.LRU,
    ) -> None:
        if eviction is not EvictionPolicy.LRU:
            raise NotImplementedError(
                f"FixedDualPool: eviction={eviction.name} is not implemented in this PR; "
                "only EvictionPolicy.LRU is wired up."
            )
        if not 0.0 < mamba_ratio < 1.0:
            raise ValueError(f"mamba_ratio must lie in the open interval (0, 1), got {mamba_ratio}")
        if attention_page_tokens <= 0:
            raise ValueError(f"attention_page_tokens must be positive, got {attention_page_tokens}")
        if total_bytes <= 0:
            raise ValueError(f"total_bytes must be positive, got {total_bytes}")

        AllocatorContext.__init__(self)
        self._model_spec = model_spec
        self._mamba_ratio = mamba_ratio
        self._dtype_bytes_f = bytes_per_element(model_spec.dtype)

        ssm_bytes = int(total_bytes * mamba_ratio)
        kv_bytes = total_bytes - ssm_bytes
        if kv_bytes <= 0 or ssm_bytes <= 0:
            raise ValueError(
                f"computed per-pool size invalid: kv_bytes={kv_bytes} ssm_bytes={ssm_bytes}"
            )

        attn_prof = model_spec.attention_profile
        ssm_prof = model_spec.ssm_profile
        per_token_kv_bytes = 2.0 * attn_prof.num_kv_heads * attn_prof.head_dim * self._dtype_bytes_f
        kv_page_size = align_up(max(1, int(per_token_kv_bytes * attention_page_tokens)))
        ssm_block_size = align_up(
            max(1, int(ssm_prof.d_inner * ssm_prof.d_state * self._dtype_bytes_f))
        )

        self._kv_store = BackingStore(total_bytes=kv_bytes, device=device)
        self._ssm_store = BackingStore(total_bytes=ssm_bytes, device=device)
        self._kv_table = PageTable(self._kv_store, page_size_bytes=kv_page_size)
        self._ssm_table = BlockTable(self._ssm_store, page_size_bytes=ssm_block_size)
        self._kv_tracker = LRURequestTracker()
        self._ssm_tracker = LRURequestTracker()

        self._handles: dict[int, PageHandle] = {}
        self._next_handle_id: int = 0
        self._pool_underused_bytes_kv: int = 0
        self._pool_underused_bytes_ssm: int = 0

    # ----- Allocator ABC -----

    def allocate(self, num_blocks: int, *, dtype_bytes: int) -> list[int]:
        if num_blocks <= 0:
            return []
        if dtype_bytes <= 0:
            raise ValueError(f"dtype_bytes must be positive, got {dtype_bytes}")
        if self._current_layer_kind is LayerKind.ATTENTION:
            return self._allocate_into(
                table=self._kv_table,
                tracker=self._kv_tracker,
                pool_id=_KV_POOL_ID,
                num_blocks=num_blocks,
            )
        return self._allocate_into(
            table=self._ssm_table,
            tracker=self._ssm_tracker,
            pool_id=_SSM_POOL_ID,
            num_blocks=num_blocks,
        )

    def free(self, block_ids: Sequence[int]) -> None:
        kv_ids: list[int] = []
        ssm_ids: list[int] = []
        for hid in block_ids:
            handle = self._handles.pop(hid, None)
            if handle is None:
                continue
            if handle.pool_id == _KV_POOL_ID:
                kv_ids.append(handle.page_id)
            else:
                ssm_ids.append(handle.page_id)
        if kv_ids:
            self._kv_table.free(kv_ids)
        if ssm_ids:
            self._ssm_table.free(ssm_ids)

    def stats(self) -> AllocatorStats:
        total = self._kv_table.num_pages_total + self._ssm_table.num_pages_total
        free = self._kv_table.num_pages_free + self._ssm_table.num_pages_free
        allocated = total - free
        ratio = (free / total) if total > 0 else 0.0
        return AllocatorStats(
            total_blocks=total,
            free_blocks=free,
            allocated_blocks=allocated,
            fragmentation_ratio=ratio,
        )

    # ----- Allocator-specific stats surface -----

    def get_allocator_stats(self) -> Mapping[str, float]:
        return {
            "pool_underused_bytes_kv": float(self._pool_underused_bytes_kv),
            "pool_underused_bytes_ssm": float(self._pool_underused_bytes_ssm),
            "mamba_ratio": float(self._mamba_ratio),
            "kv_pages_total": float(self._kv_table.num_pages_total),
            "kv_pages_used": float(self._kv_table.num_pages_used),
            "ssm_blocks_total": float(self._ssm_table.num_pages_total),
            "ssm_blocks_used": float(self._ssm_table.num_pages_used),
        }

    # ----- Helpers -----

    def _allocate_into(
        self,
        *,
        table: PageTable,
        tracker: LRURequestTracker,
        pool_id: int,
        num_blocks: int,
    ) -> list[int]:
        page_ids = self._try_alloc_with_eviction(
            table=table, tracker=tracker, num_blocks=num_blocks
        )
        handle_ids: list[int] = []
        for pid in page_ids:
            hid = self._next_handle_id
            self._next_handle_id += 1
            self._handles[hid] = PageHandle(
                page_id=pid,
                pool_id=pool_id,
                bytes_offset=table.offset_of(pid),
                bytes_size=table.page_size_bytes,
            )
            handle_ids.append(hid)
        tracker.touch(self._current_request_id, page_ids)
        return handle_ids

    def _try_alloc_with_eviction(
        self,
        *,
        table: PageTable,
        tracker: LRURequestTracker,
        num_blocks: int,
    ) -> list[int]:
        try:
            return table.alloc(num_blocks)
        except CapacityError as first:
            self._record_cross_pool_underuse(table)
            self._evict_one_from(table=table, tracker=tracker)
            try:
                return table.alloc(num_blocks)
            except CapacityError as exc:
                raise torch.cuda.OutOfMemoryError(
                    f"fixed_dual pool exhausted after eviction (origin={first}): {exc}"
                ) from exc

    def _evict_one_from(self, *, table: PageTable, tracker: LRURequestTracker) -> None:
        victim_id = tracker.select_oldest()
        if victim_id is None:
            return
        page_ids = tracker.drop(victim_id)
        # Build the inverse mapping from internal page id back to the public handle id.
        invalidated: list[int] = [
            hid for hid, handle in list(self._handles.items()) if handle.page_id in page_ids
        ]
        for hid in invalidated:
            self._handles.pop(hid, None)
        if page_ids:
            table.free(page_ids)

    def _record_cross_pool_underuse(self, target_table: PageTable) -> None:
        # When the target pool throws CapacityError, the other pool's free bytes are
        # stranded by the static partition. Add them to the running total so the
        # rigidity stat surfaces non-zero under skewed workloads.
        if target_table is self._kv_table:
            other_free = self._ssm_table.num_pages_free * self._ssm_table.page_size_bytes
            self._pool_underused_bytes_kv += other_free
        else:
            other_free = self._kv_table.num_pages_free * self._kv_table.page_size_bytes
            self._pool_underused_bytes_ssm += other_free
