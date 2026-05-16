"""Physical backing stores for AVMP.

Two thin facades, one per cache kind, each composing a
:class:`BackingStore` plus the matching table primitive from
``cachepawl.allocator.baselines.common``. The AVMP design point is
**per-pool native page sizing**: callers never pad one pool's pages up
to the other pool's size, which is the regression the vLLM unified
pool exhibits (RFC 0001 section 1, vLLM issue #37121).

Both stores expose a uniform single-page allocate/free API. The
underlying ``PageTable`` already returns 128-byte aligned page slots;
this module computes the per-pool logical size and lets the table
align internally.

FP4 (``DType.FP4``) is not wired through these stores yet. Sub-byte
packing requires layout decisions deferred to a later milestone, so
construction with an FP4 model spec raises ``NotImplementedError``
referencing RFC 0001 section 3.3 rather than silently rounding the
storage requirement down. The other supported dtypes (FP16, BF16,
INT8, FP8 variants) all have integer byte widths.
"""

from __future__ import annotations

import torch

from cachepawl.allocator.baselines.common import (
    BackingStore,
    BlockTable,
    PageTable,
)
from cachepawl.models.spec import HybridModelSpec
from cachepawl.quant.dtypes import DType, bytes_per_element


def _reject_fp4(spec: HybridModelSpec, kind: str) -> None:
    if spec.dtype is DType.FP4:
        raise NotImplementedError(
            f"{kind}: FP4 sub-byte packing is not wired through AVMP physical stores; "
            "see docs/designs/0001-asymmetric-virtual-memory-paging.md section 3.3."
        )


class KVPagesStore:
    """Fine-grained KV pages backed by one contiguous slab.

    Page size derives from the attention layer profile::

        page_size_bytes = align_up(2 * num_kv_heads * head_dim
                                   * dtype_bytes * attention_page_tokens)

    where the factor of 2 covers both keys and values. ``align_up`` runs
    inside :class:`PageTable.__init__`, so this class just supplies the
    raw byte count.
    """

    def __init__(
        self,
        model_spec: HybridModelSpec,
        attention_page_tokens: int,
        total_bytes: int,
        device: torch.device,
    ) -> None:
        if attention_page_tokens <= 0:
            raise ValueError(f"attention_page_tokens must be positive, got {attention_page_tokens}")
        _reject_fp4(model_spec, type(self).__name__)
        dtype_bytes = bytes_per_element(model_spec.dtype)
        prof = model_spec.attention_profile
        per_token = 2.0 * prof.num_kv_heads * prof.head_dim * dtype_bytes
        raw_page_bytes = max(1, int(per_token * attention_page_tokens))
        self._store = BackingStore(total_bytes=total_bytes, device=device)
        self._table = PageTable(self._store, page_size_bytes=raw_page_bytes)

    @property
    def page_size_bytes(self) -> int:
        return self._table.page_size_bytes

    @property
    def num_total(self) -> int:
        return self._table.num_pages_total

    @property
    def num_used(self) -> int:
        return self._table.num_pages_used

    @property
    def num_free(self) -> int:
        return self._table.num_pages_free

    def allocate_one(self) -> int:
        """Reserve one page and return its physical byte offset."""

        page_ids = self._table.alloc(1)
        return self._table.offset_of(page_ids[0])

    def free_one(self, physical_offset: int) -> None:
        """Return the page at ``physical_offset`` to the free list.

        Offsets handed out by :meth:`allocate_one` are exact multiples of
        :attr:`page_size_bytes`, so the inverse is integer division. A
        non-aligned offset is a caller bug and raises ``ValueError``.
        """

        page_size = self._table.page_size_bytes
        if physical_offset < 0 or physical_offset % page_size != 0:
            raise ValueError(
                f"physical_offset {physical_offset} is not a multiple of "
                f"page_size_bytes {page_size}"
            )
        page_id = physical_offset // page_size
        self._table.free([page_id])

    def resize_capacity(self, new_capacity_bytes: int) -> None:
        """Reserved for v2 sub-PR 2 (RFC 0002 section 4.4 migration mechanics)."""

        raise NotImplementedError(
            "KVPagesStore.resize_capacity: migration mechanics land in v2 sub-PR 2 "
            "(RFC 0002 section 8)"
        )


class SSMBlocksStore:
    """Coarse SSM state blocks backed by one contiguous slab.

    Block size derives from the SSM layer profile::

        block_size_bytes = align_up(d_inner * d_state * dtype_bytes)

    The block holds one full SSM state per allocation; bulk reserve and
    bulk release per sequence is the access pattern documented in RFC
    0001 section 3.1.
    """

    def __init__(
        self,
        model_spec: HybridModelSpec,
        total_bytes: int,
        device: torch.device,
    ) -> None:
        _reject_fp4(model_spec, type(self).__name__)
        dtype_bytes = bytes_per_element(model_spec.dtype)
        prof = model_spec.ssm_profile
        raw_block_bytes = max(1, int(prof.d_inner * prof.d_state * dtype_bytes))
        self._store = BackingStore(total_bytes=total_bytes, device=device)
        self._table = BlockTable(self._store, page_size_bytes=raw_block_bytes)

    @property
    def block_size_bytes(self) -> int:
        return self._table.page_size_bytes

    @property
    def num_total(self) -> int:
        return self._table.num_pages_total

    @property
    def num_used(self) -> int:
        return self._table.num_pages_used

    @property
    def num_free(self) -> int:
        return self._table.num_pages_free

    def allocate_one(self) -> int:
        """Reserve one block and return its physical byte offset."""

        block_ids = self._table.alloc(1)
        return self._table.offset_of(block_ids[0])

    def free_one(self, physical_offset: int) -> None:
        """Return the block at ``physical_offset`` to the free list."""

        block_size = self._table.page_size_bytes
        if physical_offset < 0 or physical_offset % block_size != 0:
            raise ValueError(
                f"physical_offset {physical_offset} is not a multiple of "
                f"block_size_bytes {block_size}"
            )
        block_id = physical_offset // block_size
        self._table.free([block_id])

    def resize_capacity(self, new_capacity_bytes: int) -> None:
        """Reserved for v2 sub-PR 2 (RFC 0002 section 4.4 migration mechanics)."""

        raise NotImplementedError(
            "SSMBlocksStore.resize_capacity: migration mechanics land in v2 sub-PR 2 "
            "(RFC 0002 section 8)"
        )
