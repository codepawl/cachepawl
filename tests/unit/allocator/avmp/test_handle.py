"""Unit tests for ``cachepawl.allocator.avmp.handle``.

The handle is a pure data record. Tests pin the four guarantees the
follow-up allocator relies on: value equality, hashability, frozenness,
and slot-based attribute storage.
"""

from __future__ import annotations

import dataclasses

import pytest

from cachepawl.allocator.avmp.handle import HandleKind, VirtualHandle


def _make(handle_id: int = 1, kind: HandleKind = HandleKind.KV_PAGE) -> VirtualHandle:
    return VirtualHandle(
        handle_id=handle_id,
        kind=kind,
        virtual_offset=0,
        size_bytes=65536,
        request_id="req-1",
        layer_idx=0,
    )


def test_equality_is_by_field_value() -> None:
    assert _make() == _make()
    assert _make(handle_id=1) != _make(handle_id=2)
    assert _make(kind=HandleKind.KV_PAGE) != _make(kind=HandleKind.SSM_BLOCK)


def test_hashable_so_can_live_in_a_set() -> None:
    assert {_make(), _make()} == {_make()}
    distinct = {_make(handle_id=1), _make(handle_id=2)}
    assert len(distinct) == 2


def test_frozen_blocks_attribute_assignment() -> None:
    handle = _make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        handle.handle_id = 999  # type: ignore[misc]


def test_slots_means_no_instance_dict() -> None:
    handle = _make()
    assert not hasattr(handle, "__dict__")


def test_handle_kind_has_exactly_two_members() -> None:
    names = {member.name for member in HandleKind}
    assert names == {"KV_PAGE", "SSM_BLOCK"}
