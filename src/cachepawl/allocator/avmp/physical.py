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

from cachepawl.allocator.avmp.state import ResizeResult
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
        initial_capacity_bytes: int | None = None,
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
        if initial_capacity_bytes is not None:
            if initial_capacity_bytes < 0:
                raise ValueError(
                    f"initial_capacity_bytes must be non-negative, got {initial_capacity_bytes}"
                )
            if initial_capacity_bytes > total_bytes:
                raise ValueError(
                    f"initial_capacity_bytes {initial_capacity_bytes} exceeds total_bytes "
                    f"{total_bytes}"
                )
            initial_pages = initial_capacity_bytes // self._table.page_size_bytes
            self._table.set_num_pages_total(initial_pages)

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

    def resize_capacity(self, new_capacity_bytes: int) -> ResizeResult:
        """Grow or shrink the active capacity of this store, in place.

        ``new_capacity_bytes`` is rounded DOWN to the nearest multiple of
        :attr:`page_size_bytes`. Underlying ``BackingStore`` storage is not
        re-allocated; the active page count is adjusted via
        :meth:`PageTable.set_num_pages_total` (tail-only).

        Raises ``ValueError`` if ``new_capacity_bytes`` is negative or
        exceeds the store's pre-allocated total bytes. Raises
        ``CapacityError`` (from the underlying table) if a shrink would
        leave a used page id past the new boundary.
        """

        if new_capacity_bytes < 0:
            raise ValueError(f"new_capacity_bytes must be non-negative, got {new_capacity_bytes}")
        page_size = self._table.page_size_bytes
        old_total = self._table.num_pages_total
        new_total = new_capacity_bytes // page_size
        old_capacity = old_total * page_size
        new_capacity_rounded = new_total * page_size
        self._table.set_num_pages_total(new_total)
        return ResizeResult(
            old_capacity_bytes=old_capacity,
            new_capacity_bytes=new_capacity_rounded,
            pages_delta=new_total - old_total,
            bytes_actually_moved=new_capacity_rounded - old_capacity,
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
        initial_capacity_bytes: int | None = None,
    ) -> None:
        _reject_fp4(model_spec, type(self).__name__)
        dtype_bytes = bytes_per_element(model_spec.dtype)
        prof = model_spec.ssm_profile
        raw_block_bytes = max(1, int(prof.d_inner * prof.d_state * dtype_bytes))
        self._store = BackingStore(total_bytes=total_bytes, device=device)
        self._table = BlockTable(self._store, page_size_bytes=raw_block_bytes)
        if initial_capacity_bytes is not None:
            if initial_capacity_bytes < 0:
                raise ValueError(
                    f"initial_capacity_bytes must be non-negative, got {initial_capacity_bytes}"
                )
            if initial_capacity_bytes > total_bytes:
                raise ValueError(
                    f"initial_capacity_bytes {initial_capacity_bytes} exceeds total_bytes "
                    f"{total_bytes}"
                )
            initial_blocks = initial_capacity_bytes // self._table.page_size_bytes
            self._table.set_num_pages_total(initial_blocks)

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

    def resize_capacity(self, new_capacity_bytes: int) -> ResizeResult:
        """Grow or shrink the active capacity of this store, in place.

        ``new_capacity_bytes`` is rounded DOWN to the nearest multiple of
        :attr:`block_size_bytes`. Underlying ``BackingStore`` storage is
        not re-allocated; the active block count is adjusted via
        :meth:`BlockTable.set_num_pages_total` (tail-only).

        Raises ``ValueError`` if ``new_capacity_bytes`` is negative or
        exceeds the store's pre-allocated total bytes. Raises
        ``CapacityError`` (from the underlying table) if a shrink would
        leave a used block id past the new boundary.
        """

        if new_capacity_bytes < 0:
            raise ValueError(f"new_capacity_bytes must be non-negative, got {new_capacity_bytes}")
        block_size = self._table.page_size_bytes
        old_total = self._table.num_pages_total
        new_total = new_capacity_bytes // block_size
        old_capacity = old_total * block_size
        new_capacity_rounded = new_total * block_size
        self._table.set_num_pages_total(new_total)
        return ResizeResult(
            old_capacity_bytes=old_capacity,
            new_capacity_bytes=new_capacity_rounded,
            pages_delta=new_total - old_total,
            bytes_actually_moved=new_capacity_rounded - old_capacity,
        )
