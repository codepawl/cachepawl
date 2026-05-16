"""Observe-only contract tests for AVMP v2.

When ``rebalance_enabled=False`` the pool reproduces v1 behavior: all 14
v2 stats keys are present at their default values, ``kv_free_ratio`` and
``ssm_free_ratio`` track live allocation, and no migration runs.

When ``rebalance_enabled=True`` AND the pool comes under pressure, sub-PR
3's auto-trigger fires off the observation hook. The "state observed but
not acted upon" assertions from sub-PR 1 are obsolete in this PR; the
new "auto-trigger fires under pressure" contract is tested in
``test_avmp_v2_auto_trigger.py``.
"""

from __future__ import annotations

import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
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
    # No migration runs without an explicit trigger.
    assert stats["rebalance_count"] == 0.0
    assert stats["bytes_migrated_total"] == 0.0
    assert stats["auto_rebalance_skipped_throttle"] == 0.0


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
    assert stats["bytes_wasted_to_alignment_total"] == 0.0
    assert stats["auto_rebalance_skipped_throttle"] == 0.0
    # Empty pool: full free ratios.
    assert stats["kv_free_ratio"] == 1.0
    assert stats["ssm_free_ratio"] == 1.0
    assert stats["current_kv_pool_bytes"] == stats["kv_pool_bytes"]
    assert stats["current_ssm_pool_bytes"] == stats["ssm_pool_bytes"]


def test_rebalance_disabled_remains_byte_silent_under_load(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """With ``rebalance_enabled=False``, draining one side leaves capacity
    unchanged: no auto-trigger, no manual trigger, no surprises."""

    pool = _make_pool(jamba_spec, cpu_device, rebalance_enabled=False)
    baseline = pool.get_allocator_stats()
    kv_total = int(baseline["kv_pages_total"])
    ssm_total = int(baseline["ssm_blocks_total"])

    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    pool.allocate(kv_total, dtype_bytes=2)
    pool.set_current_layer_kind(LayerKind.MAMBA2)
    pool.set_current_request_id(2)
    pool.allocate(ssm_total // 2, dtype_bytes=2)

    stats = pool.get_allocator_stats()
    # Capacity unchanged.
    assert stats["current_kv_pool_bytes"] == baseline["kv_pool_bytes"]
    assert stats["current_ssm_pool_bytes"] == baseline["ssm_pool_bytes"]
    # Migration counters all zero.
    assert stats["rebalance_count"] == 0.0
    assert stats["bytes_migrated_total"] == 0.0
    assert stats["time_spent_rebalancing_ns"] == 0.0
    assert stats["bytes_wasted_to_alignment_total"] == 0.0
    assert stats["auto_rebalance_skipped_throttle"] == 0.0
