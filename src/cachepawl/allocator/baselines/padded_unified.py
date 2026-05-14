"""vLLM-style padded unified KV pool.

Mirrors the page-size unification performed by vLLM's
``unify_kv_cache_spec_page_size`` at commit
``40330967ab8e718c186e99bb08cbd3b65281e396``:
https://github.com/vllm-project/vllm/blob/40330967ab8e718c186e99bb08cbd3b65281e396/vllm/v1/core/kv_cache_utils.py#L1009

In a hybrid model that interleaves attention and SSM layers vLLM unifies
the per-layer page size by inflating every smaller page up to the
maximum. The formula in upstream is::

    max_page_size = max(layer.page_size_bytes for layer in groups)
    ratio = max_page_size // layer_page_size
    new_block_size = layer_spec.block_size * ratio

In this baseline the same effect is reproduced by computing one
canonical ``page_size_bytes`` from
``max(attention_page_bytes, ssm_block_bytes)``. Allocations whose
logical bytes are smaller than the page size therefore waste
``(page_size_bytes - logical_bytes) * num_blocks`` per call. The
running total of that waste over all live pages is reported as
``padding_waste_bytes`` in :meth:`get_allocator_stats`; it is the
quantity that vLLM issue
`#37121 <https://github.com/vllm-project/vllm/issues/37121>`_ reports as
13.7 percent utilization on Qwen3.5-4B-AWQ.

The pool also honours the documented OOM contract: when the page table
is exhausted the LRU request is dropped and its pages are reclaimed; if
no evictable request exists, :class:`torch.cuda.OutOfMemoryError` is
raised. The same exception class works on CPU and CUDA.

Only :class:`EvictionPolicy.LRU` is implemented in this PR. Passing any
other policy at construction raises ``NotImplementedError`` with a clear
message naming what is unwired.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from cachepawl.allocator.base import Allocator, AllocatorStats
from cachepawl.allocator.baselines.common import (
    AllocatorContext,
    BackingStore,
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


class PaddedUnifiedPool(Allocator, AllocatorContext):
    """Padded unified KV pool that mirrors vLLM's HybridKVCacheCoordinator."""

    def __init__(
        self,
        model_spec: HybridModelSpec,
        total_bytes: int,
        device: torch.device,
        attention_page_tokens: int = 16,
        eviction: EvictionPolicy = EvictionPolicy.LRU,
    ) -> None:
        if eviction is not EvictionPolicy.LRU:
            raise NotImplementedError(
                f"PaddedUnifiedPool: eviction={eviction.name} is not implemented in this PR; "
                "only EvictionPolicy.LRU is wired up."
            )
        if attention_page_tokens <= 0:
            raise ValueError(f"attention_page_tokens must be positive, got {attention_page_tokens}")
        AllocatorContext.__init__(self)
        self._model_spec = model_spec
        self._attention_page_tokens = attention_page_tokens
        self._dtype_bytes_f = bytes_per_element(model_spec.dtype)

        attn_bytes = self._attention_page_bytes_raw()
        ssm_bytes = self._ssm_block_bytes_raw()
        unified_page_size = align_up(max(attn_bytes, ssm_bytes))

        self._page_size_bytes = unified_page_size
        self._attention_logical_bytes = attn_bytes
        self._ssm_logical_bytes = ssm_bytes

        self._store = BackingStore(total_bytes=total_bytes, device=device)
        self._table = PageTable(self._store, page_size_bytes=unified_page_size)
        self._tracker = LRURequestTracker()
        self._handles: dict[int, PageHandle] = {}
        self._padding_waste_bytes: int = 0

    # ----- Allocator ABC -----

    def allocate(self, num_blocks: int, *, dtype_bytes: int) -> list[int]:
        if num_blocks <= 0:
            return []
        if dtype_bytes <= 0:
            raise ValueError(f"dtype_bytes must be positive, got {dtype_bytes}")
        try:
            page_ids = self._table.alloc(num_blocks)
        except CapacityError:
            self._evict_one_request()
            try:
                page_ids = self._table.alloc(num_blocks)
            except CapacityError as exc:
                raise torch.cuda.OutOfMemoryError(
                    f"padded_unified pool exhausted after eviction: {exc}"
                ) from exc

        logical_bytes_per_page = self._logical_bytes_for_current_kind()
        waste_per_page = max(0, self._page_size_bytes - logical_bytes_per_page)
        self._padding_waste_bytes += waste_per_page * num_blocks

        for pid in page_ids:
            self._handles[pid] = PageHandle(
                page_id=pid,
                pool_id=_KV_POOL_ID,
                bytes_offset=self._table.offset_of(pid),
                bytes_size=logical_bytes_per_page,
            )
        self._tracker.touch(self._current_request_id, page_ids)
        return page_ids

    def free(self, block_ids: Sequence[int]) -> None:
        seen: set[int] = set()
        to_free: list[int] = []
        for pid in block_ids:
            if pid in seen or pid not in self._handles:
                continue
            seen.add(pid)
            to_free.append(pid)
        if not to_free:
            return
        for pid in to_free:
            handle = self._handles.pop(pid)
            waste = max(0, self._page_size_bytes - handle.bytes_size)
            self._padding_waste_bytes = max(0, self._padding_waste_bytes - waste)
        self._table.free(to_free)
        self._tracker.remove_pages(to_free)

    def stats(self) -> AllocatorStats:
        total = self._table.num_pages_total
        free = self._table.num_pages_free
        allocated = total - free
        ratio = (free / total) if total > 0 else 0.0
        return AllocatorStats(
            total_blocks=total,
            free_blocks=free,
            allocated_blocks=allocated,
            fragmentation_ratio=ratio,
        )

    # ----- Allocator-specific stats surface (read by the runner via duck typing) -----

    def get_allocator_stats(self) -> Mapping[str, float]:
        return {
            "padding_waste_bytes": float(self._padding_waste_bytes),
            "num_pages_total": float(self._table.num_pages_total),
            "num_pages_used": float(self._table.num_pages_used),
            "page_size_bytes": float(self._page_size_bytes),
        }

    # ----- Helpers -----

    def _attention_page_bytes_raw(self) -> int:
        prof = self._model_spec.attention_profile
        per_token = 2.0 * prof.num_kv_heads * prof.head_dim * self._dtype_bytes_f
        return max(1, int(per_token * self._attention_page_tokens))

    def _ssm_block_bytes_raw(self) -> int:
        prof = self._model_spec.ssm_profile
        return max(1, int(prof.d_inner * prof.d_state * self._dtype_bytes_f))

    def _logical_bytes_for_current_kind(self) -> int:
        if self._current_layer_kind is LayerKind.ATTENTION:
            return self._attention_logical_bytes
        return self._ssm_logical_bytes

    def _evict_one_request(self) -> None:
        victim_id = self._tracker.select_oldest()
        if victim_id is None:
            return
        page_ids = self._tracker.drop(victim_id)
        for pid in page_ids:
            handle = self._handles.pop(pid, None)
            if handle is not None:
                waste = max(0, self._page_size_bytes - handle.bytes_size)
                self._padding_waste_bytes = max(0, self._padding_waste_bytes - waste)
        if page_ids:
            self._table.free(page_ids)
