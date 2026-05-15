"""Virtual page table for AVMP.

Maps minted ``handle_id`` integers to ``(VirtualHandle, physical_offset)``
pairs, with one internal dict per :class:`HandleKind` so the per-kind
live-count queries stay O(1). The two-dict layout is intentional: in
the follow-up pool implementation, ``num_kv_handles_live`` and
``num_ssm_handles_live`` feed allocator stats sampled at every tick,
and an O(n) scan over all live handles per sample would dominate the
profile under load.

Handle ids are minted from a single monotonically-increasing counter
shared across both kinds and **never reused** within one
:class:`VirtualPageTable` instance. The reason: handle ids are not VRAM
bytes, so non-reuse costs nothing real, and it keeps a stale handle's
``resolve`` raising rather than silently colliding with an unrelated
fresh allocation. A future generation-counter scheme (RFC 0001 section
3.1) can layer on top once mid-sequence rebalancing lands; the v1
prototype gets by with the simpler invariant.

The pool that owns this table dereferences ``handle_id`` into a
``(VirtualHandle, physical_offset)`` pair via :meth:`resolve`. On free
it calls :meth:`remove` to retrieve the physical offset back so the
matching store can reclaim the page or block.
"""

from __future__ import annotations

from dataclasses import dataclass

from cachepawl.allocator.avmp.handle import HandleKind, VirtualHandle

INVALID_HANDLE_ID: int = 0


@dataclass(frozen=True, slots=True)
class _Entry:
    virtual_handle: VirtualHandle
    physical_offset: int


class VirtualPageTable:
    """Two-dict page table keyed by handle id, partitioned by kind."""

    def __init__(self) -> None:
        self._kv_entries: dict[int, _Entry] = {}
        self._ssm_entries: dict[int, _Entry] = {}
        self._next_handle_id: int = INVALID_HANDLE_ID + 1
        self._total_virtual_bytes_live: int = 0

    @property
    def num_kv_handles_live(self) -> int:
        return len(self._kv_entries)

    @property
    def num_ssm_handles_live(self) -> int:
        return len(self._ssm_entries)

    @property
    def total_virtual_bytes_live(self) -> int:
        return self._total_virtual_bytes_live

    def mint(
        self,
        kind: HandleKind,
        virtual_offset: int,
        size_bytes: int,
        request_id: str,
        layer_idx: int,
        physical_offset: int,
    ) -> VirtualHandle:
        """Mint a new :class:`VirtualHandle` and store its mapping.

        Raises ``ValueError`` for non-positive ``size_bytes`` or negative
        offsets; the table does not silently accept zero-byte allocations
        because they break the live-bytes accounting and are never what
        the caller meant.
        """

        if size_bytes <= 0:
            raise ValueError(f"size_bytes must be positive, got {size_bytes}")
        if virtual_offset < 0:
            raise ValueError(f"virtual_offset must be non-negative, got {virtual_offset}")
        if physical_offset < 0:
            raise ValueError(f"physical_offset must be non-negative, got {physical_offset}")
        handle_id = self._next_handle_id
        self._next_handle_id += 1
        handle = VirtualHandle(
            handle_id=handle_id,
            kind=kind,
            virtual_offset=virtual_offset,
            size_bytes=size_bytes,
            request_id=request_id,
            layer_idx=layer_idx,
        )
        entry = _Entry(virtual_handle=handle, physical_offset=physical_offset)
        if kind is HandleKind.KV_PAGE:
            self._kv_entries[handle_id] = entry
        else:
            self._ssm_entries[handle_id] = entry
        self._total_virtual_bytes_live += size_bytes
        return handle

    def resolve(self, handle_id: int) -> tuple[VirtualHandle, int]:
        """Look up ``handle_id`` and return ``(handle, physical_offset)``.

        Raises ``KeyError`` if the id has been freed or was never minted.
        Checking both dicts is O(1) per lookup; the kind partition is for
        live-count speed, not lookup speed.
        """

        entry = self._kv_entries.get(handle_id)
        if entry is None:
            entry = self._ssm_entries.get(handle_id)
        if entry is None:
            raise KeyError(f"handle_id {handle_id} is not live")
        return entry.virtual_handle, entry.physical_offset

    def remove(self, handle_id: int) -> tuple[VirtualHandle, int]:
        """Drop ``handle_id`` from the table and return its prior entry.

        Raises ``KeyError`` if the id is not live. The caller is the only
        component that knows which physical store to return the offset to,
        so :class:`VirtualPageTable` itself does not call ``free_one`` on
        any store.
        """

        entry = self._kv_entries.pop(handle_id, None)
        if entry is None:
            entry = self._ssm_entries.pop(handle_id, None)
        if entry is None:
            raise KeyError(f"handle_id {handle_id} is not live")
        self._total_virtual_bytes_live -= entry.virtual_handle.size_bytes
        return entry.virtual_handle, entry.physical_offset
