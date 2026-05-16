"""Byte-identical determinism contract for AVMP v2 with rebalance_enabled=True.

PR #13's wall-clock throttle (``time.monotonic_ns``) made auto-trigger
decisions wall-clock dependent, so the same workload produced different
aggregated stats across runs (concrete RFC 0002 section 8 question 5).

v2 sub-PR 4 replaced the wall clock with a logical operation counter on
:class:`PoolPressureMonitor`. This test runs an identical allocate / free
trajectory through two pool instances and asserts the aggregated stats
agree byte-for-byte (modulo intrinsically wall-clock fields like
``time_spent_rebalancing_ns``).
"""

from __future__ import annotations

import contextlib

import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.models.spec import HybridModelSpec, LayerKind

_TOTAL_16_MIB = 16 * 1024 * 1024
# Wall-clock fields excluded from the byte-identicality assertion. Everything
# else must match exactly across runs.
_NON_DETERMINISTIC_FIELDS: frozenset[str] = frozenset({"time_spent_rebalancing_ns"})


def _drive_workload(pool: AsymmetricVirtualPool) -> None:
    """Fixed allocate / free trajectory that crosses pressure thresholds.

    Mixes KV-only and SSM-only requests, drains and frees in interleaved
    rounds. Designed to hit the CapacityError branch (and therefore the
    auto-trigger) several times.
    """

    kv_total = int(pool.get_allocator_stats()["kv_pages_total"])
    ssm_total = int(pool.get_allocator_stats()["ssm_blocks_total"])

    # Round 1: drain KV with two requests then over-allocate.
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    pool.allocate(kv_total // 2, dtype_bytes=2)
    pool.set_current_request_id(2)
    pool.allocate(kv_total // 2, dtype_bytes=2)
    pool.set_current_request_id(3)
    with contextlib.suppress(torch.cuda.OutOfMemoryError):
        pool.allocate(kv_total, dtype_bytes=2)

    # Round 2: drain SSM the same way.
    pool.set_current_layer_kind(LayerKind.MAMBA2)
    pool.set_current_request_id(4)
    pool.allocate(ssm_total // 2, dtype_bytes=2)
    pool.set_current_request_id(5)
    pool.allocate(ssm_total // 2, dtype_bytes=2)
    pool.set_current_request_id(6)
    with contextlib.suppress(torch.cuda.OutOfMemoryError):
        pool.allocate(ssm_total, dtype_bytes=2)


def _run(spec: HybridModelSpec, device: torch.device) -> dict[str, float]:
    pool = AsymmetricVirtualPool(
        model_spec=spec,
        total_bytes=_TOTAL_16_MIB,
        device=device,
        mamba_ratio=0.5,
        rebalance_enabled=True,
        min_rebalance_interval_ops=0,
    )
    _drive_workload(pool)
    return dict(pool.get_allocator_stats())


def test_two_identical_runs_produce_byte_identical_stats(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """RFC 0002 section 8 question 5 fix: throttle uses operation_count, so
    decisions are deterministic across runs."""

    a = _run(jamba_spec, cpu_device)
    b = _run(jamba_spec, cpu_device)
    for key in _NON_DETERMINISTIC_FIELDS:
        a.pop(key, None)
        b.pop(key, None)
    assert a == b


def test_critical_counters_match_exactly(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Counter fields documented in the brief MUST match exactly:
    rebalance_count, bytes_migrated_total, current_kv_pool_bytes,
    current_ssm_pool_bytes, auto_rebalance_skipped_throttle,
    bytes_wasted_to_alignment_total.
    """

    a = _run(jamba_spec, cpu_device)
    b = _run(jamba_spec, cpu_device)
    for key in (
        "rebalance_count",
        "bytes_migrated_total",
        "bytes_wasted_to_alignment_total",
        "current_kv_pool_bytes",
        "current_ssm_pool_bytes",
        "auto_rebalance_skipped_throttle",
        "current_pressure_state_code",
        "kv_free_ratio",
        "ssm_free_ratio",
    ):
        assert a[key] == b[key], f"{key} drifted: {a[key]} vs {b[key]}"


def test_workload_actually_exercises_rebalance(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Guards against silent regression: the determinism test would be
    vacuous if the workload never triggered a rebalance."""

    stats = _run(jamba_spec, cpu_device)
    assert stats["rebalance_count"] >= 1.0
