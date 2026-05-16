"""End-to-end tests for AVMP v2 sub-PR 2 migration mechanics.

Exercises ``trigger_manual_rebalance`` via ``_apply_rebalance``: the
happy paths in both directions, page-size-rounding waste accounting,
rejection on insufficient source capacity, rollback on recipient grow
failure, the diagnostic-use case when ``rebalance_enabled=False``, and
determinism of ``RebalanceOutcome`` across identical starting states.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.allocator.avmp.state import (
    RebalanceDirection,
    RebalanceOutcome,
    ResizeResult,
)
from cachepawl.allocator.baselines.common import CapacityError
from cachepawl.models.spec import HybridModelSpec, LayerKind

_TOTAL_64_MIB = 64 * 1024 * 1024


def _make_pool(
    spec: HybridModelSpec,
    device: torch.device,
    *,
    rebalance_enabled: bool = True,
) -> AsymmetricVirtualPool:
    return AsymmetricVirtualPool(
        model_spec=spec,
        total_bytes=_TOTAL_64_MIB,
        device=device,
        mamba_ratio=0.5,
        rebalance_enabled=rebalance_enabled,
    )


def test_ssm_to_kv_migration_shifts_capacity_and_records_waste(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    before = pool.get_allocator_stats()
    page_size = pool._kv_store.page_size_bytes
    block_size = pool._ssm_store.block_size_bytes

    outcome = pool.trigger_manual_rebalance(RebalanceDirection.SSM_TO_KV, batch_blocks=2)

    expected_donor_delta = 2 * block_size
    expected_recipient_grow = (expected_donor_delta // page_size) * page_size
    expected_waste = expected_donor_delta - expected_recipient_grow

    assert outcome.success
    assert outcome.failure_reason is None
    assert outcome.bytes_migrated == expected_donor_delta
    assert outcome.bytes_wasted_to_alignment == expected_waste
    assert outcome.elapsed_ns > 0

    after = pool.get_allocator_stats()
    assert (
        after["current_ssm_pool_bytes"] == before["current_ssm_pool_bytes"] - expected_donor_delta
    )
    assert (
        after["current_kv_pool_bytes"] == before["current_kv_pool_bytes"] + expected_recipient_grow
    )
    assert after["rebalance_count"] == 1.0
    assert after["bytes_migrated_total"] == float(expected_donor_delta)
    assert after["bytes_wasted_to_alignment_total"] == float(expected_waste)
    assert after["time_spent_rebalancing_ns"] > 0
    # Conservation: kv + ssm + cumulative waste equals the initial pool sum.
    assert (
        after["current_kv_pool_bytes"]
        + after["current_ssm_pool_bytes"]
        + after["bytes_wasted_to_alignment_total"]
        == before["current_kv_pool_bytes"] + before["current_ssm_pool_bytes"]
    )


def test_kv_to_ssm_migration_mirror(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    before = pool.get_allocator_stats()
    page_size = pool._kv_store.page_size_bytes
    block_size = pool._ssm_store.block_size_bytes

    outcome = pool.trigger_manual_rebalance(RebalanceDirection.KV_TO_SSM, batch_blocks=2)

    expected_donor_delta = 2 * page_size
    expected_recipient_grow = (expected_donor_delta // block_size) * block_size
    expected_waste = expected_donor_delta - expected_recipient_grow

    assert outcome.success
    assert outcome.bytes_migrated == expected_donor_delta
    assert outcome.bytes_wasted_to_alignment == expected_waste

    after = pool.get_allocator_stats()
    assert after["current_kv_pool_bytes"] == before["current_kv_pool_bytes"] - expected_donor_delta
    assert (
        after["current_ssm_pool_bytes"]
        == before["current_ssm_pool_bytes"] + expected_recipient_grow
    )


def test_migration_rejected_when_donor_has_insufficient_free_capacity(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Drain the donor; the next trigger should report success=False with
    pool sizes unchanged."""

    pool = _make_pool(jamba_spec, cpu_device)
    # Drain SSM completely. SSM_TO_KV migration then has no free block to surrender.
    ssm_total = int(pool.get_allocator_stats()["ssm_blocks_total"])
    pool.set_current_layer_kind(LayerKind.MAMBA2)
    pool.set_current_request_id(1)
    pool.allocate(ssm_total, dtype_bytes=2)
    before = pool.get_allocator_stats()

    outcome = pool.trigger_manual_rebalance(RebalanceDirection.SSM_TO_KV, batch_blocks=1)

    assert not outcome.success
    assert outcome.failure_reason is not None
    assert "donor shrink rejected" in outcome.failure_reason

    after = pool.get_allocator_stats()
    assert after["current_kv_pool_bytes"] == before["current_kv_pool_bytes"]
    assert after["current_ssm_pool_bytes"] == before["current_ssm_pool_bytes"]
    assert after["rebalance_count"] == 0.0
    assert after["bytes_migrated_total"] == 0.0


def test_migration_rolls_back_donor_when_recipient_grow_fails(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recipient failure path: donor must be restored, counters must not move."""

    pool = _make_pool(jamba_spec, cpu_device)
    before = pool.get_allocator_stats()

    real_kv_resize = pool._kv_store.resize_capacity
    call_count = {"n": 0}

    def _failing_then_passing(new_capacity_bytes: int) -> ResizeResult:
        # First call is the recipient grow (SSM_TO_KV direction). Fail it.
        # Subsequent calls (from sub-PR 3 etc.) would pass through, but only
        # the test's one trigger should fire while this monkeypatch is active.
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise CapacityError("simulated recipient failure")
        return real_kv_resize(new_capacity_bytes)

    monkeypatch.setattr(pool._kv_store, "resize_capacity", _failing_then_passing)

    outcome = pool.trigger_manual_rebalance(RebalanceDirection.SSM_TO_KV, batch_blocks=2)

    assert not outcome.success
    assert outcome.failure_reason is not None
    assert "recipient grow rejected" in outcome.failure_reason

    after = pool.get_allocator_stats()
    # Donor must be back to its pre-migration size.
    assert after["current_ssm_pool_bytes"] == before["current_ssm_pool_bytes"]
    # Recipient unchanged from the start (failed before commit).
    assert after["current_kv_pool_bytes"] == before["current_kv_pool_bytes"]
    # Counters did not move.
    assert after["rebalance_count"] == 0.0
    assert after["bytes_migrated_total"] == 0.0
    assert after["bytes_wasted_to_alignment_total"] == 0.0


def test_manual_trigger_works_when_rebalance_enabled_is_false(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Diagnostic use case: counters update even though the monitor is absent.

    Pressure state stays at BALANCED throughout because the monitor (which
    is the source of transition records) was never constructed.
    """

    pool = _make_pool(jamba_spec, cpu_device, rebalance_enabled=False)
    before = pool.get_allocator_stats()
    assert before["rebalance_enabled"] == 0.0
    assert pool._pressure_monitor is None

    outcome = pool.trigger_manual_rebalance(RebalanceDirection.SSM_TO_KV, batch_blocks=1)

    assert outcome.success
    after = pool.get_allocator_stats()
    assert after["rebalance_count"] == 1.0
    assert after["bytes_migrated_total"] > 0.0
    # Monitor is None so no transition was recorded; pool stays in BALANCED.
    assert after["current_pressure_state_code"] == 0.0  # BALANCED


def test_trigger_manual_rebalance_rejects_zero_or_negative_batch_blocks(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    with pytest.raises(ValueError, match="batch_blocks must be >= 1"):
        pool.trigger_manual_rebalance(RebalanceDirection.SSM_TO_KV, batch_blocks=0)
    with pytest.raises(ValueError, match="batch_blocks must be >= 1"):
        pool.trigger_manual_rebalance(RebalanceDirection.SSM_TO_KV, batch_blocks=-1)


def test_rebalance_outcome_is_deterministic_across_identical_pools(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Same starting state + same trigger -> identical outcome (excluding elapsed_ns)."""

    pool_a = _make_pool(jamba_spec, cpu_device)
    pool_b = _make_pool(jamba_spec, cpu_device)

    out_a = pool_a.trigger_manual_rebalance(RebalanceDirection.SSM_TO_KV, batch_blocks=2)
    out_b = pool_b.trigger_manual_rebalance(RebalanceDirection.SSM_TO_KV, batch_blocks=2)

    # Compare every field except elapsed_ns (timing-dependent).
    canonical_a = replace(out_a, elapsed_ns=0)
    canonical_b = replace(out_b, elapsed_ns=0)
    assert canonical_a == canonical_b


def test_rebalance_outcome_dataclass_is_frozen_and_slots(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    outcome = pool.trigger_manual_rebalance(RebalanceDirection.SSM_TO_KV, batch_blocks=1)
    assert isinstance(outcome, RebalanceOutcome)
    with pytest.raises(AttributeError):
        outcome.success = False  # type: ignore[misc]


def test_multiple_migrations_accumulate_counters(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    block_size = pool._ssm_store.block_size_bytes
    page_size = pool._kv_store.page_size_bytes

    for _ in range(3):
        pool.trigger_manual_rebalance(RebalanceDirection.SSM_TO_KV, batch_blocks=1)

    stats = pool.get_allocator_stats()
    assert stats["rebalance_count"] == 3.0
    assert stats["bytes_migrated_total"] == float(3 * block_size)
    expected_waste_per_event = block_size - (block_size // page_size) * page_size
    assert stats["bytes_wasted_to_alignment_total"] == float(3 * expected_waste_per_event)
