"""Behavioral tests for :class:`AsymmetricVirtualPool`.

Covers construction validation, single-pool and mixed-pool
allocate/free roundtrips, same-pool LRU eviction, and the
cross-pool isolation contract that AVMP v1 must hold to (filling
one pool while the other has free bytes raises OOM rather than
triggering an eviction into the other pool).
"""

from __future__ import annotations

import pytest
import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.allocator.policy import EvictionPolicy
from cachepawl.models.spec import HybridModelSpec, LayerKind

_TOTAL_64_MIB = 64 * 1024 * 1024
_TOTAL_4_MIB = 4 * 1024 * 1024


def _make_pool(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
    *,
    total_bytes: int = _TOTAL_64_MIB,
    mamba_ratio: float = 0.5,
) -> AsymmetricVirtualPool:
    return AsymmetricVirtualPool(
        model_spec=jamba_spec,
        total_bytes=total_bytes,
        device=cpu_device,
        mamba_ratio=mamba_ratio,
    )


def test_construct_with_neutral_ratio_splits_bytes_roughly_evenly(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device, mamba_ratio=0.5)
    stats = pool.get_allocator_stats()
    assert stats["kv_pool_bytes"] > 0
    assert stats["ssm_pool_bytes"] > 0
    # Sum is bounded by total_bytes from above; one full page or block of
    # slack on each side is acceptable due to alignment rounding.
    assert stats["kv_pool_bytes"] + stats["ssm_pool_bytes"] <= _TOTAL_64_MIB
    assert stats["kv_pool_bytes"] + stats["ssm_pool_bytes"] >= _TOTAL_64_MIB - 2 * 262144


def test_construct_with_asymmetric_ratio_skews_toward_kv(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device, mamba_ratio=0.2)
    stats = pool.get_allocator_stats()
    assert stats["kv_pool_bytes"] > stats["ssm_pool_bytes"]
    # mamba_ratio=0.2 means SSM gets ~20% of total; KV gets ~80%.
    assert stats["kv_pool_bytes"] > 0.7 * _TOTAL_64_MIB
    assert stats["ssm_pool_bytes"] < 0.3 * _TOTAL_64_MIB


@pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.1, 1.1])
def test_mamba_ratio_extremes_rejected(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
    bad_ratio: float,
) -> None:
    with pytest.raises(ValueError, match="mamba_ratio"):
        _make_pool(jamba_spec, cpu_device, mamba_ratio=bad_ratio)


def test_non_lru_eviction_policy_raises(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    with pytest.raises(NotImplementedError, match=r"FIFO"):
        AsymmetricVirtualPool(
            model_spec=jamba_spec,
            total_bytes=_TOTAL_4_MIB,
            device=cpu_device,
            eviction=EvictionPolicy.FIFO,
        )


def test_zero_total_bytes_rejected(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    with pytest.raises(ValueError, match="total_bytes"):
        AsymmetricVirtualPool(
            model_spec=jamba_spec,
            total_bytes=0,
            device=cpu_device,
        )


def test_allocate_zero_blocks_returns_empty(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    assert pool.allocate(0, dtype_bytes=2) == []


def test_allocate_rejects_non_positive_dtype_bytes(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    with pytest.raises(ValueError, match="dtype_bytes"):
        pool.allocate(1, dtype_bytes=0)


def test_kv_allocate_free_roundtrip(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    ids = pool.allocate(3, dtype_bytes=2)
    assert len(ids) == 3
    stats_after_alloc = pool.get_allocator_stats()
    assert stats_after_alloc["kv_pages_used"] == 3.0
    assert stats_after_alloc["ssm_blocks_used"] == 0.0
    assert stats_after_alloc["virtual_handles_live"] == 3.0

    pool.free(ids)
    stats_after_free = pool.get_allocator_stats()
    assert stats_after_free["kv_pages_used"] == 0.0
    assert stats_after_free["virtual_handles_live"] == 0.0


def test_ssm_allocate_free_roundtrip(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    pool.set_current_layer_kind(LayerKind.MAMBA2)
    pool.set_current_request_id(1)
    ids = pool.allocate(2, dtype_bytes=2)
    assert len(ids) == 2
    stats_after_alloc = pool.get_allocator_stats()
    assert stats_after_alloc["ssm_blocks_used"] == 2.0
    assert stats_after_alloc["kv_pages_used"] == 0.0

    pool.free(ids)
    stats_after_free = pool.get_allocator_stats()
    assert stats_after_free["ssm_blocks_used"] == 0.0
    assert stats_after_free["virtual_handles_live"] == 0.0


def test_mixed_pool_free_in_one_call(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    kv_ids = pool.allocate(4, dtype_bytes=2)

    pool.set_current_layer_kind(LayerKind.MAMBA2)
    pool.set_current_request_id(2)
    ssm_ids = pool.allocate(3, dtype_bytes=2)

    pool.free(kv_ids + ssm_ids)

    stats = pool.get_allocator_stats()
    assert stats["kv_pages_used"] == 0.0
    assert stats["ssm_blocks_used"] == 0.0
    assert stats["virtual_handles_live"] == 0.0


def test_same_pool_lru_eviction(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device, total_bytes=_TOTAL_4_MIB)
    kv_total = int(pool.get_allocator_stats()["kv_pages_total"])

    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    request_1_ids = pool.allocate(kv_total, dtype_bytes=2)
    assert pool.get_allocator_stats()["kv_pages_free"] == 0.0

    pool.set_current_request_id(2)
    request_2_ids = pool.allocate(1, dtype_bytes=2)
    assert len(request_2_ids) == 1

    stats = pool.get_allocator_stats()
    assert stats["kv_pages_used"] == 1.0
    assert stats["virtual_handles_live"] == 1.0

    # Request 1's handles must no longer resolve; pool.free should
    # silently no-op on every one.
    pool.free(request_1_ids)
    stats_after = pool.get_allocator_stats()
    assert stats_after["kv_pages_used"] == 1.0
    assert stats_after["virtual_handles_live"] == 1.0


def test_cross_pool_eviction_is_disabled(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """KV exhaustion must NOT evict SSM requests, even under OOM pressure."""

    pool = _make_pool(jamba_spec, cpu_device, total_bytes=_TOTAL_4_MIB)
    kv_total = int(pool.get_allocator_stats()["kv_pages_total"])

    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    pool.allocate(kv_total, dtype_bytes=2)

    pool.set_current_request_id(2)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        pool.allocate(kv_total + 1, dtype_bytes=2)

    stats = pool.get_allocator_stats()
    assert stats["ssm_blocks_used"] == 0.0
    assert stats["cross_pool_eviction_count"] == 0.0


def test_free_unknown_handle_is_idempotent(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    pool.free([999999, 888888])  # never minted
    stats = pool.get_allocator_stats()
    assert stats["kv_pages_used"] == 0.0
    assert stats["ssm_blocks_used"] == 0.0


def test_free_dedups_duplicate_ids(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    ids = pool.allocate(2, dtype_bytes=2)
    pool.free([ids[0], ids[0], ids[1]])  # double-free in one call
    stats = pool.get_allocator_stats()
    assert stats["kv_pages_used"] == 0.0


def test_stats_top_level_shape(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    pool.allocate(2, dtype_bytes=2)
    snapshot = pool.stats()
    assert snapshot.allocated_blocks == 2
    assert snapshot.free_blocks == snapshot.total_blocks - 2
    assert 0.0 <= snapshot.fragmentation_ratio <= 1.0
