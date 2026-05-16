"""Unit tests for KVPagesStore / SSMBlocksStore resize_capacity.

Replaces the deleted test_resize_capacity_stub.py from sub-PR 1. Exercises
grow, shrink, the page-size rounding-down rule, and the CapacityError raised
when shrinking past in-use pages.
"""

from __future__ import annotations

import pytest
import torch

from cachepawl.allocator.avmp.physical import KVPagesStore, SSMBlocksStore
from cachepawl.allocator.avmp.state import ResizeResult
from cachepawl.allocator.baselines.common import CapacityError
from cachepawl.models.spec import HybridModelSpec


def _make_kv_store(
    spec: HybridModelSpec,
    device: torch.device,
    *,
    total_bytes: int,
    initial_capacity_bytes: int | None = None,
) -> KVPagesStore:
    return KVPagesStore(
        model_spec=spec,
        attention_page_tokens=16,
        total_bytes=total_bytes,
        device=device,
        initial_capacity_bytes=initial_capacity_bytes,
    )


def _make_ssm_store(
    spec: HybridModelSpec,
    device: torch.device,
    *,
    total_bytes: int,
    initial_capacity_bytes: int | None = None,
) -> SSMBlocksStore:
    return SSMBlocksStore(
        model_spec=spec,
        total_bytes=total_bytes,
        device=device,
        initial_capacity_bytes=initial_capacity_bytes,
    )


def test_kv_resize_capacity_grow_returns_result_and_increases_total(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = _make_kv_store(
        jamba_spec, cpu_device, total_bytes=4 * 1024 * 1024, initial_capacity_bytes=1024 * 1024
    )
    page_size = store.page_size_bytes
    old_total = store.num_total

    result = store.resize_capacity(2 * 1024 * 1024)

    assert isinstance(result, ResizeResult)
    assert result.old_capacity_bytes == old_total * page_size
    assert result.new_capacity_bytes == (2 * 1024 * 1024 // page_size) * page_size
    assert result.pages_delta > 0
    assert result.bytes_actually_moved > 0
    assert store.num_total > old_total


def test_kv_resize_capacity_shrink_returns_result_and_decreases_total(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = _make_kv_store(jamba_spec, cpu_device, total_bytes=4 * 1024 * 1024)
    page_size = store.page_size_bytes
    old_total = store.num_total

    result = store.resize_capacity(1 * 1024 * 1024)

    assert result.new_capacity_bytes == (1024 * 1024 // page_size) * page_size
    assert result.pages_delta < 0
    assert result.bytes_actually_moved < 0
    assert store.num_total < old_total


def test_kv_resize_capacity_rounds_new_capacity_down_to_page_multiple(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = _make_kv_store(
        jamba_spec, cpu_device, total_bytes=4 * 1024 * 1024, initial_capacity_bytes=0
    )
    page_size = store.page_size_bytes
    # Ask for 1.5 * page_size; should round down to one page.
    odd_request = page_size + page_size // 2

    result = store.resize_capacity(odd_request)

    assert result.new_capacity_bytes == page_size
    assert store.num_total == 1


def test_kv_resize_capacity_shrink_rejected_when_used_page_in_tail(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = _make_kv_store(jamba_spec, cpu_device, total_bytes=4 * 1024 * 1024)
    # Consume one page; id 0 is now used.
    used_offset = store.allocate_one()
    assert used_offset == 0
    # Shrinking to zero pages would strand the used id 0.
    with pytest.raises(CapacityError, match="still in use"):
        store.resize_capacity(0)


def test_kv_resize_capacity_to_zero_succeeds_when_idle(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = _make_kv_store(jamba_spec, cpu_device, total_bytes=4 * 1024 * 1024)

    result = store.resize_capacity(0)

    assert result.new_capacity_bytes == 0
    assert store.num_total == 0
    assert store.num_used == 0


def test_kv_resize_capacity_negative_rejected(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = _make_kv_store(jamba_spec, cpu_device, total_bytes=4 * 1024 * 1024)
    with pytest.raises(ValueError, match="non-negative"):
        store.resize_capacity(-1)


def test_kv_resize_capacity_beyond_pre_allocated_max_rejected(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = _make_kv_store(jamba_spec, cpu_device, total_bytes=4 * 1024 * 1024)
    # BackingStore allocated 4 MiB; growing to 8 MiB exceeds capacity.
    with pytest.raises(ValueError, match="exceeds backing store capacity"):
        store.resize_capacity(8 * 1024 * 1024)


def test_ssm_resize_capacity_grow_and_shrink_round_trip(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = _make_ssm_store(
        jamba_spec, cpu_device, total_bytes=4 * 1024 * 1024, initial_capacity_bytes=1024 * 1024
    )
    block_size = store.block_size_bytes
    initial_total = store.num_total

    grow = store.resize_capacity(2 * 1024 * 1024)
    assert grow.pages_delta > 0
    assert store.num_total > initial_total

    shrink = store.resize_capacity(512 * 1024)
    assert shrink.pages_delta < 0
    assert store.num_total == 512 * 1024 // block_size


def test_ssm_resize_capacity_shrink_rejected_when_used_block_in_tail(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = _make_ssm_store(jamba_spec, cpu_device, total_bytes=4 * 1024 * 1024)
    store.allocate_one()  # used id 0
    with pytest.raises(CapacityError, match="still in use"):
        store.resize_capacity(0)


def test_ssm_resize_capacity_rounds_down(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = _make_ssm_store(
        jamba_spec, cpu_device, total_bytes=4 * 1024 * 1024, initial_capacity_bytes=0
    )
    block_size = store.block_size_bytes
    odd_request = block_size + block_size // 2

    result = store.resize_capacity(odd_request)

    assert result.new_capacity_bytes == block_size
    assert store.num_total == 1


def test_resize_result_dataclass_is_frozen_and_slots(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = _make_kv_store(jamba_spec, cpu_device, total_bytes=4 * 1024 * 1024)
    result = store.resize_capacity(1024 * 1024)
    with pytest.raises(AttributeError):
        result.old_capacity_bytes = 999  # type: ignore[misc]
