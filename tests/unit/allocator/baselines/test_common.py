"""Unit tests for :mod:`cachepawl.allocator.baselines.common`.

Currently exercises ``PageTable.set_num_pages_total`` because that is the
new API surface added in AVMP v2 sub-PR 2 (RFC 0002 section 4.4). The
method is AVMP-only; baselines never call it.
"""

from __future__ import annotations

import pytest
import torch

from cachepawl.allocator.baselines.common import (
    BackingStore,
    BlockTable,
    CapacityError,
    PageTable,
    align_up,
)


def _make_table(total_bytes: int, page_size_bytes: int) -> PageTable:
    store = BackingStore(total_bytes=total_bytes, device=torch.device("cpu"))
    return PageTable(store, page_size_bytes=page_size_bytes)


def test_set_num_pages_total_shrink_drops_trailing_free_ids() -> None:
    table = _make_table(total_bytes=10 * 128, page_size_bytes=128)
    assert table.num_pages_total == 10
    # Allocate 3 pages (ids 0, 1, 2 by alloc-from-end-of-reverse-list).
    used = table.alloc(3)
    assert used == [0, 1, 2]

    table.set_num_pages_total(5)

    assert table.num_pages_total == 5
    # Used ids 0,1,2 still allocated; remaining free ids in [3, 4].
    assert table.num_pages_used == 3
    assert sorted(table.alloc(2)) == [3, 4]


def test_set_num_pages_total_grow_appends_new_high_ids() -> None:
    table = _make_table(total_bytes=10 * 128, page_size_bytes=128)
    # Shrink first to give grow something to undo.
    table.set_num_pages_total(4)
    assert table.num_pages_total == 4

    table.set_num_pages_total(7)

    assert table.num_pages_total == 7
    assert table.num_pages_free == 7
    # All seven ids are now allocatable.
    ids = sorted(table.alloc(7))
    assert ids == [0, 1, 2, 3, 4, 5, 6]


def test_set_num_pages_total_shrink_rejected_when_used_page_in_tail() -> None:
    table = _make_table(total_bytes=10 * 128, page_size_bytes=128)
    table.alloc(8)  # uses ids 0..7
    with pytest.raises(CapacityError, match="page id 7 is still in use"):
        table.set_num_pages_total(5)


def test_set_num_pages_total_grow_beyond_backing_store_capacity_raises() -> None:
    table = _make_table(total_bytes=10 * 128, page_size_bytes=128)
    with pytest.raises(ValueError, match="exceeds backing store capacity"):
        table.set_num_pages_total(11)


def test_set_num_pages_total_to_zero_when_idle() -> None:
    table = _make_table(total_bytes=10 * 128, page_size_bytes=128)

    table.set_num_pages_total(0)

    assert table.num_pages_total == 0
    assert table.num_pages_free == 0
    assert table.num_pages_used == 0
    with pytest.raises(CapacityError):
        table.alloc(1)


def test_set_num_pages_total_to_zero_rejected_when_pages_used() -> None:
    table = _make_table(total_bytes=10 * 128, page_size_bytes=128)
    table.alloc(1)  # uses id 0
    with pytest.raises(CapacityError, match="page id 0 is still in use"):
        table.set_num_pages_total(0)


def test_set_num_pages_total_negative_rejected() -> None:
    table = _make_table(total_bytes=10 * 128, page_size_bytes=128)
    with pytest.raises(ValueError, match="new_total must be non-negative"):
        table.set_num_pages_total(-1)


def test_set_num_pages_total_idempotent_at_current_value() -> None:
    table = _make_table(total_bytes=10 * 128, page_size_bytes=128)
    table.alloc(3)
    before_free = list(table._free_pages)
    table.set_num_pages_total(table.num_pages_total)
    assert table.num_pages_total == 10
    assert list(table._free_pages) == before_free


def test_block_table_inherits_set_num_pages_total() -> None:
    store = BackingStore(total_bytes=10 * 128, device=torch.device("cpu"))
    block_table = BlockTable(store, page_size_bytes=128)
    assert block_table.num_pages_total == 10
    block_table.set_num_pages_total(4)
    assert block_table.num_pages_total == 4


def test_set_num_pages_total_respects_page_size_alignment() -> None:
    """page_size_bytes is align_up'd; max capacity uses the aligned value."""

    page_size = 96
    aligned = align_up(page_size)  # 128 with PAGE_ALIGNMENT_BYTES=128
    store = BackingStore(total_bytes=5 * aligned, device=torch.device("cpu"))
    table = PageTable(store, page_size_bytes=page_size)
    assert table.page_size_bytes == aligned
    # 5 * 128 / 128 = 5 pages max
    table.set_num_pages_total(5)
    assert table.num_pages_total == 5
    with pytest.raises(ValueError, match="exceeds backing store capacity"):
        table.set_num_pages_total(6)
