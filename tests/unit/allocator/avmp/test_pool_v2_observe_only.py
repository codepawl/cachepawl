"""Observe-only behavior tests for AVMP v2 sub-PR 1.

When ``rebalance_enabled=True`` the pool's ``PoolPressureMonitor`` records
state transitions, but physical pool sizes never change and migration
counters stay at zero. When ``rebalance_enabled=False`` all twelve new
stats keys are still present, defaulting to neutral values, and the live
ratios still update on allocate / free.
"""

from __future__ import annotations

import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.allocator.avmp.state import PoolPressureState
from cachepawl.models.spec import HybridModelSpec, LayerKind

_TOTAL_16_MIB = 16 * 1024 * 1024


def _make_pool(
    spec: HybridModelSpec,
    device: torch.device,
    *,
    rebalance_enabled: bool,
) -> AsymmetricVirtualPool:
    return AsymmetricVirtualPool(
        model_spec=spec,
        total_bytes=_TOTAL_16_MIB,
        device=device,
        mamba_ratio=0.5,
        rebalance_enabled=rebalance_enabled,
    )


def test_kv_pressure_drives_state_code_to_kv_pressured(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device, rebalance_enabled=True)
    baseline = pool.get_allocator_stats()
    kv_total = int(baseline["kv_pages_total"])

    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    pool.allocate(kv_total, dtype_bytes=2)

    stats = pool.get_allocator_stats()
    assert stats["current_pressure_state_code"] == float(PoolPressureState.KV_PRESSURED.value)
    assert stats["kv_free_ratio"] == 0.0
    assert stats["ssm_free_ratio"] == 1.0
    # Pool sizes are still static in sub-PR 1.
    assert stats["current_kv_pool_bytes"] == baseline["kv_pool_bytes"]
    assert stats["current_ssm_pool_bytes"] == baseline["ssm_pool_bytes"]
    # Migration counters stay zero in this PR.
    assert stats["rebalance_count"] == 0.0
    assert stats["bytes_migrated_total"] == 0.0
    assert stats["time_spent_rebalancing_ns"] == 0.0


def test_ssm_pressure_drives_state_code_to_ssm_pressured(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device, rebalance_enabled=True)
    baseline = pool.get_allocator_stats()
    ssm_total = int(baseline["ssm_blocks_total"])

    pool.set_current_layer_kind(LayerKind.MAMBA2)
    pool.set_current_request_id(1)
    pool.allocate(ssm_total, dtype_bytes=2)

    stats = pool.get_allocator_stats()
    assert stats["current_pressure_state_code"] == float(PoolPressureState.SSM_PRESSURED.value)
    assert stats["ssm_free_ratio"] == 0.0
    assert stats["kv_free_ratio"] == 1.0
    assert stats["current_kv_pool_bytes"] == baseline["kv_pool_bytes"]
    assert stats["current_ssm_pool_bytes"] == baseline["ssm_pool_bytes"]
    assert stats["rebalance_count"] == 0.0
    assert stats["bytes_migrated_total"] == 0.0
    assert stats["time_spent_rebalancing_ns"] == 0.0


def test_free_returns_state_to_balanced(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """A KV-pressured state must clear when the holding request is freed."""

    pool = _make_pool(jamba_spec, cpu_device, rebalance_enabled=True)
    kv_total = int(pool.get_allocator_stats()["kv_pages_total"])
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    ids = pool.allocate(kv_total, dtype_bytes=2)
    assert pool.get_allocator_stats()["current_pressure_state_code"] == float(
        PoolPressureState.KV_PRESSURED.value
    )

    pool.free(ids)
    assert pool.get_allocator_stats()["current_pressure_state_code"] == float(
        PoolPressureState.BALANCED.value
    )


def test_rebalance_disabled_keeps_state_code_at_balanced(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """With ``rebalance_enabled=False`` the monitor is absent; state stays 0."""

    pool = _make_pool(jamba_spec, cpu_device, rebalance_enabled=False)
    kv_total = int(pool.get_allocator_stats()["kv_pages_total"])
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    pool.allocate(kv_total, dtype_bytes=2)

    stats = pool.get_allocator_stats()
    assert stats["rebalance_enabled"] == 0.0
    assert stats["current_pressure_state_code"] == 0.0  # BALANCED
    # Ratios still update even when the monitor is off.
    assert stats["kv_free_ratio"] == 0.0
    assert stats["ssm_free_ratio"] == 1.0


def test_rebalance_disabled_surfaces_all_v2_keys_at_defaults(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device, rebalance_enabled=False)
    stats = pool.get_allocator_stats()
    assert stats["rebalance_enabled"] == 0.0
    assert stats["threshold_low"] == 0.05
    assert stats["threshold_high"] == 0.30
    assert stats["migration_batch_size"] == 1.0
    assert stats["current_pressure_state_code"] == 0.0
    assert stats["rebalance_count"] == 0.0
    assert stats["bytes_migrated_total"] == 0.0
    assert stats["time_spent_rebalancing_ns"] == 0.0
    # Empty pool: full free ratios.
    assert stats["kv_free_ratio"] == 1.0
    assert stats["ssm_free_ratio"] == 1.0
    assert stats["current_kv_pool_bytes"] == stats["kv_pool_bytes"]
    assert stats["current_ssm_pool_bytes"] == stats["ssm_pool_bytes"]
