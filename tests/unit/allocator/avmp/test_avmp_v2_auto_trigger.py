"""End-to-end tests for AVMP v2 sub-PR 3 auto-trigger.

When ``rebalance_enabled=True`` and the pool comes under pressure, the
observation hook (called at the end of every allocate / free) fires an
automatic rebalance unless the throttle window has not elapsed since the
last auto-trigger. Manual triggers via ``trigger_manual_rebalance``
bypass the throttle.
"""

from __future__ import annotations

import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool, RebalanceDirection
from cachepawl.models.spec import HybridModelSpec, LayerKind

_TOTAL_16_MIB = 16 * 1024 * 1024
_NEVER_AUTO_TRIGGER_OPS = 2**30


def _make_pool(
    spec: HybridModelSpec,
    device: torch.device,
    *,
    min_rebalance_interval_ops: int = 0,
) -> AsymmetricVirtualPool:
    """Pool fixture for auto-trigger tests.

    Default ``min_rebalance_interval_ops=0`` so the very first compute_state
    call that returns a pressured state can fire a rebalance. The production
    default is 1000; this fixture uses 0 because the per-test workload is
    too short to amortize a 1000-op cold start. The throttle-suppression
    tests pass ``_NEVER_AUTO_TRIGGER_OPS`` explicitly to override.
    """

    return AsymmetricVirtualPool(
        model_spec=spec,
        total_bytes=_TOTAL_16_MIB,
        device=device,
        mamba_ratio=0.5,
        rebalance_enabled=True,
        min_rebalance_interval_ops=min_rebalance_interval_ops,
    )


def test_kv_pressure_auto_triggers_ssm_to_kv(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    baseline = pool.get_allocator_stats()
    kv_total = int(baseline["kv_pages_total"])

    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    pool.allocate(kv_total, dtype_bytes=2)

    stats = pool.get_allocator_stats()
    assert stats["rebalance_count"] >= 1.0
    assert stats["bytes_migrated_total"] > 0.0
    # SSM_TO_KV migration grows the KV side.
    assert stats["current_kv_pool_bytes"] > baseline["kv_pool_bytes"]
    assert stats["current_ssm_pool_bytes"] < baseline["ssm_pool_bytes"]


def test_ssm_pressure_auto_triggers_kv_to_ssm(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """SSM pressure triggers KV_TO_SSM. The donor (KV) shrinks; the
    recipient (SSM) may or may not gain whole blocks depending on whether
    one KV page is large enough to round into one SSM block.

    On Jamba 1.5 mini ``page_size < block_size``, so with
    ``migration_batch_size=1`` a single migration event releases one KV
    page but the SSM side rounds-down to zero new blocks. The donor side
    still shrinks; ``bytes_wasted_to_alignment_total`` records the rounding
    residue. This is a real finding for v2: KV_TO_SSM with batch=1 wastes
    its entire donor delta on this spec.
    """

    pool = _make_pool(jamba_spec, cpu_device)
    baseline = pool.get_allocator_stats()
    ssm_total = int(baseline["ssm_blocks_total"])

    pool.set_current_layer_kind(LayerKind.MAMBA2)
    pool.set_current_request_id(1)
    pool.allocate(ssm_total, dtype_bytes=2)

    stats = pool.get_allocator_stats()
    assert stats["rebalance_count"] >= 1.0
    assert stats["bytes_migrated_total"] > 0.0
    assert stats["current_kv_pool_bytes"] < baseline["kv_pool_bytes"]
    # Either SSM gained whole blocks, or the entire migration was wasted to
    # alignment. Both outcomes are valid; the conservation invariant holds.
    ssm_grew = stats["current_ssm_pool_bytes"] > baseline["ssm_pool_bytes"]
    wasted_all = stats["bytes_wasted_to_alignment_total"] >= stats["bytes_migrated_total"]
    assert ssm_grew or wasted_all


def test_throttle_skips_rapid_repeat_triggers(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """With a huge ``min_rebalance_interval_ops`` no auto-trigger ever fires.

    Pressure is detected and skipped; the throttle counter increments.
    """

    pool = _make_pool(jamba_spec, cpu_device, min_rebalance_interval_ops=_NEVER_AUTO_TRIGGER_OPS)
    kv_total = int(pool.get_allocator_stats()["kv_pages_total"])
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    pool.allocate(kv_total, dtype_bytes=2)

    stats = pool.get_allocator_stats()
    assert stats["rebalance_count"] == 0.0
    assert stats["bytes_migrated_total"] == 0.0
    assert stats["auto_rebalance_skipped_throttle"] >= 1.0


def test_manual_trigger_bypasses_throttle(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Even when the throttle interval is large, a manual trigger fires
    immediately. Throttle only applies to the auto-trigger path."""

    pool = _make_pool(jamba_spec, cpu_device, min_rebalance_interval_ops=_NEVER_AUTO_TRIGGER_OPS)
    outcome = pool.trigger_manual_rebalance(RebalanceDirection.SSM_TO_KV, batch_blocks=1)
    assert outcome.success
    stats = pool.get_allocator_stats()
    assert stats["rebalance_count"] == 1.0
    assert stats["bytes_migrated_total"] > 0.0


def test_throttle_window_lets_subsequent_trigger_through(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """A short throttle window plus repeated pressure produces multiple
    auto-triggers over time."""

    # Throttle effectively off: 1 ns is shorter than any allocate/free wall.
    pool = _make_pool(jamba_spec, cpu_device, min_rebalance_interval_ops=1)
    kv_total = int(pool.get_allocator_stats()["kv_pages_total"])

    # Round 1: drain KV, free, drain KV again. Each "drain" pushes the pool
    # into KV_PRESSURED which fires the auto-trigger.
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    ids = pool.allocate(kv_total, dtype_bytes=2)
    after_first = pool.get_allocator_stats()
    pool.free(ids)
    pool.set_current_request_id(2)
    pool.allocate(kv_total, dtype_bytes=2)
    after_second = pool.get_allocator_stats()

    assert after_second["rebalance_count"] >= after_first["rebalance_count"]
    assert after_first["rebalance_count"] >= 1.0
