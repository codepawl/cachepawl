"""Unit tests for ``cachepawl.allocator.avmp.page_table``.

Pins the invariants the follow-up allocator relies on:

- ``mint`` produces strictly-ascending unique handle ids and never
  reuses an id even after removal (use-after-free safety).
- ``resolve`` and ``remove`` partition lookup correctness, raising
  ``KeyError`` on missing ids rather than silently returning a default.
- Per-kind live counts and ``total_virtual_bytes_live`` stay consistent
  with the actual mint/remove history under random ops.
- Construction-time validation rejects zero-byte or negative-offset
  mints rather than corrupting the bytes-live counter.
"""

from __future__ import annotations

import random

import pytest

from cachepawl.allocator.avmp.handle import HandleKind
from cachepawl.allocator.avmp.page_table import INVALID_HANDLE_ID, VirtualPageTable


def _mint_kv(table: VirtualPageTable, size_bytes: int = 4096) -> int:
    handle = table.mint(
        kind=HandleKind.KV_PAGE,
        virtual_offset=0,
        size_bytes=size_bytes,
        request_id="req",
        layer_idx=0,
        physical_offset=0,
    )
    return handle.handle_id


def _mint_ssm(table: VirtualPageTable, size_bytes: int = 262144) -> int:
    handle = table.mint(
        kind=HandleKind.SSM_BLOCK,
        virtual_offset=0,
        size_bytes=size_bytes,
        request_id="req",
        layer_idx=1,
        physical_offset=0,
    )
    return handle.handle_id


def test_mint_returns_strictly_ascending_unique_ids() -> None:
    table = VirtualPageTable()
    ids = [_mint_kv(table) for _ in range(5)]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    assert all(handle_id != INVALID_HANDLE_ID for handle_id in ids)


def test_mint_ids_are_shared_across_kinds() -> None:
    table = VirtualPageTable()
    kv_first = _mint_kv(table)
    ssm_next = _mint_ssm(table)
    assert ssm_next > kv_first


def test_resolve_returns_matching_entry() -> None:
    table = VirtualPageTable()
    handle = table.mint(
        kind=HandleKind.KV_PAGE,
        virtual_offset=128,
        size_bytes=65536,
        request_id="req-7",
        layer_idx=3,
        physical_offset=65536,
    )
    resolved_handle, resolved_offset = table.resolve(handle.handle_id)
    assert resolved_handle == handle
    assert resolved_offset == 65536


def test_remove_makes_subsequent_resolve_raise() -> None:
    table = VirtualPageTable()
    handle_id = _mint_kv(table)
    table.remove(handle_id)
    with pytest.raises(KeyError):
        table.resolve(handle_id)


def test_remove_of_missing_id_raises() -> None:
    table = VirtualPageTable()
    with pytest.raises(KeyError):
        table.remove(123456)


def test_handle_ids_are_never_reused_after_removal() -> None:
    table = VirtualPageTable()
    first = _mint_kv(table)
    table.remove(first)
    second = _mint_kv(table)
    assert second > first


def test_mint_rejects_zero_size() -> None:
    table = VirtualPageTable()
    with pytest.raises(ValueError, match="size_bytes"):
        table.mint(
            kind=HandleKind.KV_PAGE,
            virtual_offset=0,
            size_bytes=0,
            request_id="req",
            layer_idx=0,
            physical_offset=0,
        )


def test_mint_rejects_negative_offsets() -> None:
    table = VirtualPageTable()
    with pytest.raises(ValueError, match="virtual_offset"):
        table.mint(
            kind=HandleKind.KV_PAGE,
            virtual_offset=-1,
            size_bytes=4096,
            request_id="req",
            layer_idx=0,
            physical_offset=0,
        )
    with pytest.raises(ValueError, match="physical_offset"):
        table.mint(
            kind=HandleKind.SSM_BLOCK,
            virtual_offset=0,
            size_bytes=4096,
            request_id="req",
            layer_idx=0,
            physical_offset=-1,
        )


def test_random_mint_remove_keeps_stats_consistent() -> None:
    table = VirtualPageTable()
    rng = random.Random(42)
    live_kv: dict[int, int] = {}  # handle_id -> size_bytes
    live_ssm: dict[int, int] = {}
    for _ in range(100):
        live_total = len(live_kv) + len(live_ssm)
        if live_total > 0 and rng.random() < 0.4:
            pool = live_kv if rng.random() < 0.5 and live_kv else live_ssm
            if not pool:
                pool = live_kv if live_kv else live_ssm
            handle_id = rng.choice(list(pool.keys()))
            pool.pop(handle_id)
            table.remove(handle_id)
        else:
            size = rng.choice([4096, 65536, 262144])
            if rng.random() < 0.5:
                live_kv[_mint_kv(table, size_bytes=size)] = size
            else:
                live_ssm[_mint_ssm(table, size_bytes=size)] = size
        assert table.num_kv_handles_live == len(live_kv)
        assert table.num_ssm_handles_live == len(live_ssm)
        assert table.total_virtual_bytes_live == sum(live_kv.values()) + sum(live_ssm.values())
