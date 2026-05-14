"""Shared primitives for the baseline allocators.

Both ``PaddedUnifiedPool`` and ``FixedDualPool`` build on a small set of
data structures that own a contiguous torch buffer, hand out fixed-size
page slots, and track per-request page ownership for LRU eviction. The
implementations are intentionally simple; the baselines exist to expose
each upstream system's documented weakness, not to be fast.

Real ``torch`` tensors back every byte the allocator hands out so that
``torch.cuda.memory_stats`` reports realistic numbers on CUDA. On CPU
the same calls produce host-memory tensors and the harness still works.

Page-size alignment: every byte size returned to a caller is rounded up
to the nearest 128 bytes via :func:`align_up`. The 128-byte multiple is
the typical Triton coalesced-load width and keeps the eventual real
kernels happy; if strict alignment proves unnecessary later, the multi
ple can drop without changing the public surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter_ns

import torch

from cachepawl.models.spec import LayerKind

PAGE_ALIGNMENT_BYTES: int = 128


class CapacityError(Exception):
    """Raised when a pool has no free pages left."""


def align_up(num_bytes: int, multiple: int = PAGE_ALIGNMENT_BYTES) -> int:
    """Round ``num_bytes`` up to the nearest multiple of ``multiple``."""

    if multiple <= 0:
        raise ValueError(f"alignment must be positive, got {multiple}")
    if num_bytes <= 0:
        return multiple
    return ((num_bytes + multiple - 1) // multiple) * multiple


@dataclass(frozen=True, slots=True)
class PageHandle:
    """Internal record describing one allocated page.

    External callers only ever see ``page_id`` (returned as an ``int``
    from :meth:`Allocator.allocate`). The handle is kept inside the
    allocator so it can compute padding waste and per-pool occupancy
    without a second pass.
    """

    page_id: int
    pool_id: int
    bytes_offset: int
    bytes_size: int


class BackingStore:
    """Single ``torch.uint8`` buffer plus a byte-range view factory.

    ``view(offset, size)`` returns a 1-D ``uint8`` slice into the
    underlying buffer. The store does not free the underlying tensor
    until it is garbage collected.
    """

    def __init__(self, total_bytes: int, device: torch.device) -> None:
        if total_bytes <= 0:
            raise ValueError(f"total_bytes must be positive, got {total_bytes}")
        self._device = device
        self._total_bytes = total_bytes
        self._buffer = torch.empty(total_bytes, dtype=torch.uint8, device=device)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def device(self) -> torch.device:
        return self._device

    def view(self, offset: int, size: int) -> torch.Tensor:
        if offset < 0 or size <= 0 or offset + size > self._total_bytes:
            raise ValueError(
                f"invalid view range: offset={offset} size={size} total={self._total_bytes}"
            )
        return self._buffer.narrow(0, offset, size)


class PageTable:
    """Fixed-size paging over a ``BackingStore`` slab.

    Allocates pages in 128-byte aligned chunks from a free list. The
    table owns a contiguous region of the store starting at
    ``start_offset`` of length ``page_size_bytes * num_pages``.
    """

    def __init__(
        self,
        store: BackingStore,
        page_size_bytes: int,
        start_offset: int = 0,
    ) -> None:
        if page_size_bytes <= 0:
            raise ValueError(f"page_size_bytes must be positive, got {page_size_bytes}")
        aligned = align_up(page_size_bytes)
        usable = store.total_bytes - start_offset
        if usable <= 0:
            raise ValueError(f"start_offset {start_offset} exceeds store size {store.total_bytes}")
        self._store = store
        self._page_size = aligned
        self._start_offset = start_offset
        self._num_pages_total = usable // aligned
        self._free_pages: list[int] = list(range(self._num_pages_total - 1, -1, -1))

    @property
    def page_size_bytes(self) -> int:
        return self._page_size

    @property
    def num_pages_total(self) -> int:
        return self._num_pages_total

    @property
    def num_pages_free(self) -> int:
        return len(self._free_pages)

    @property
    def num_pages_used(self) -> int:
        return self._num_pages_total - len(self._free_pages)

    def alloc(self, n: int) -> list[int]:
        if n <= 0:
            return []
        if n > len(self._free_pages):
            raise CapacityError(
                f"requested {n} pages but only {len(self._free_pages)} are free "
                f"(total {self._num_pages_total})"
            )
        out: list[int] = []
        for _ in range(n):
            out.append(self._free_pages.pop())
        return out

    def free(self, ids: Sequence[int]) -> None:
        for pid in ids:
            if not 0 <= pid < self._num_pages_total:
                raise ValueError(f"page id {pid} out of range [0, {self._num_pages_total})")
            self._free_pages.append(pid)

    def offset_of(self, page_id: int) -> int:
        if not 0 <= page_id < self._num_pages_total:
            raise ValueError(f"page id {page_id} out of range [0, {self._num_pages_total})")
        return self._start_offset + page_id * self._page_size


class BlockTable(PageTable):
    """Coarse-block paging table.

    Identical surface to ``PageTable``; the distinct name marks the
    semantic role of holding one fixed-size SSM state per allocation in
    :class:`FixedDualPool`. Kept as a subclass so that future SSM-only
    behavior (refcount, copy-on-write) can land here without touching
    KV-side code.
    """


@dataclass(slots=True)
class _RequestEntry:
    last_access_ns: int
    page_ids: list[int] = field(default_factory=list)


class LRURequestTracker:
    """Maps request id to its allocated page ids and last access time.

    The allocator owns eviction policy; this tracker only ranks the
    requests by recency and surrenders page ids on drop.
    """

    def __init__(self) -> None:
        self._entries: dict[int, _RequestEntry] = {}

    def touch(self, request_id: int, page_ids: Sequence[int]) -> None:
        entry = self._entries.get(request_id)
        if entry is None:
            self._entries[request_id] = _RequestEntry(
                last_access_ns=perf_counter_ns(),
                page_ids=list(page_ids),
            )
            return
        entry.last_access_ns = perf_counter_ns()
        entry.page_ids.extend(page_ids)

    def select_oldest(self) -> int | None:
        if not self._entries:
            return None
        return min(self._entries.items(), key=lambda kv: kv[1].last_access_ns)[0]

    def drop(self, request_id: int) -> list[int]:
        entry = self._entries.pop(request_id, None)
        if entry is None:
            return []
        return entry.page_ids

    def page_ids_for(self, request_id: int) -> list[int]:
        entry = self._entries.get(request_id)
        return [] if entry is None else list(entry.page_ids)

    def remove_pages(self, page_ids: Sequence[int]) -> None:
        """Remove ``page_ids`` from every entry, dropping now-empty entries.

        Called whenever a public ``Allocator.free`` returns pages to a
        table so that subsequent ``select_oldest`` / ``drop`` calls do
        not see stale page ids that have already been freed. Without
        this, eviction can pick a departed request's lingering entry
        and re-free its pages, growing ``num_pages_free`` past
        ``num_pages_total``.
        """

        if not page_ids:
            return
        page_set = set(page_ids)
        empty_keys: list[int] = []
        for req_id, entry in self._entries.items():
            entry.page_ids = [pid for pid in entry.page_ids if pid not in page_set]
            if not entry.page_ids:
                empty_keys.append(req_id)
        for req_id in empty_keys:
            del self._entries[req_id]

    def __len__(self) -> int:
        return len(self._entries)


class AllocatorContext:
    """Mixin granting baseline allocators per-allocate context.

    The :class:`Allocator` ABC has no parameter for layer kind or owning
    request id, so baselines that need either read thread-local values
    that the runner sets immediately before each ``allocate`` call.

    - ``current_layer_kind`` routes ``FixedDualPool`` to the KV or SSM
      sub-pool. ``PaddedUnifiedPool`` ignores it.
    - ``current_request_id`` is used by both baselines to group pages
      for LRU eviction. Defaults to ``-1`` when the runner has not set
      anything, which puts every page in one anonymous bucket.
    """

    def __init__(self) -> None:
        self._current_layer_kind: LayerKind = LayerKind.ATTENTION
        self._current_request_id: int = -1

    def set_current_layer_kind(self, kind: LayerKind) -> None:
        self._current_layer_kind = kind

    def set_current_request_id(self, request_id: int) -> None:
        self._current_request_id = request_id

    @property
    def current_layer_kind(self) -> LayerKind:
        return self._current_layer_kind

    @property
    def current_request_id(self) -> int:
        return self._current_request_id
