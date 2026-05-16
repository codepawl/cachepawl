"""Stats schema and sanity-invariant tests for AsymmetricVirtualPool.

The pool surfaces its kind-specific stats via ``get_allocator_stats``,
which the runner reads through the duck-typed
``_AllocatorStatsExporter`` protocol. The mapping must be strictly
``str -> float`` (the schema's contract for
``AllocatorMetrics.allocator_specific_stats``). String identifiers
such as the allocator kind belong at ``BenchmarkRun.allocator_name``,
not in this mapping; the harness sets that field from the registry
key when the pool is registered in a follow-up PR.
"""

from __future__ import annotations

import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.benchmarks import (
    PRESETS,
    AllocatorMetrics,
    BenchmarkRun,
    Environment,
    Hardware,
)
from cachepawl.models.spec import HybridModelSpec, LayerKind

_EXPECTED_V1_KEYS: frozenset[str] = frozenset(
    {
        "kv_pages_total",
        "kv_pages_used",
        "kv_pages_free",
        "ssm_blocks_total",
        "ssm_blocks_used",
        "ssm_blocks_free",
        "virtual_handles_live",
        "cross_pool_eviction_count",
        "kv_pool_bytes",
        "ssm_pool_bytes",
        "mamba_ratio",
    }
)

# v2 sub-PR 1 introduced twelve observability keys; sub-PR 2 adds the
# thirteenth, ``bytes_wasted_to_alignment_total``, which accumulates the
# page-size rounding residue across successful migrations.
_EXPECTED_V2_KEYS: frozenset[str] = frozenset(
    {
        "rebalance_enabled",
        "threshold_low",
        "threshold_high",
        "migration_batch_size",
        "current_kv_pool_bytes",
        "current_ssm_pool_bytes",
        "kv_free_ratio",
        "ssm_free_ratio",
        "current_pressure_state_code",
        "rebalance_count",
        "bytes_migrated_total",
        "time_spent_rebalancing_ns",
        "bytes_wasted_to_alignment_total",
    }
)

_EXPECTED_KEYS: frozenset[str] = _EXPECTED_V1_KEYS | _EXPECTED_V2_KEYS


def _make_pool(spec: HybridModelSpec, device: torch.device) -> AsymmetricVirtualPool:
    return AsymmetricVirtualPool(
        model_spec=spec,
        total_bytes=4 * 1024 * 1024,
        device=device,
        mamba_ratio=0.5,
    )


def _drive_mixed_workload(pool: AsymmetricVirtualPool) -> tuple[list[int], list[int]]:
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    kv_ids = pool.allocate(3, dtype_bytes=2)
    pool.set_current_layer_kind(LayerKind.MAMBA2)
    pool.set_current_request_id(2)
    ssm_ids = pool.allocate(2, dtype_bytes=2)
    return kv_ids, ssm_ids


def test_get_allocator_stats_returns_exactly_the_documented_keys(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    _drive_mixed_workload(pool)
    stats = pool.get_allocator_stats()
    assert frozenset(stats.keys()) == _EXPECTED_KEYS


def test_every_stat_value_is_a_float(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    _drive_mixed_workload(pool)
    stats = pool.get_allocator_stats()
    for key, value in stats.items():
        assert isinstance(value, float), f"{key} is {type(value).__name__}, not float"


def test_pool_sanity_invariants_hold_after_mixed_workload(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    _drive_mixed_workload(pool)
    stats = pool.get_allocator_stats()
    assert stats["kv_pages_used"] + stats["kv_pages_free"] == stats["kv_pages_total"]
    assert stats["ssm_blocks_used"] + stats["ssm_blocks_free"] == stats["ssm_blocks_total"]
    assert stats["virtual_handles_live"] == stats["kv_pages_used"] + stats["ssm_blocks_used"]
    assert stats["cross_pool_eviction_count"] == 0.0


def test_cross_pool_eviction_count_stays_zero_through_eviction(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """v1 contract: no cross-pool eviction event is ever recorded.

    Force a same-pool eviction and check the counter still reads 0.
    The field exists so v2 can light it up without a schema bump.
    """

    pool = _make_pool(jamba_spec, cpu_device)
    kv_total = int(pool.get_allocator_stats()["kv_pages_total"])
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    pool.allocate(kv_total, dtype_bytes=2)
    pool.set_current_request_id(2)
    pool.allocate(1, dtype_bytes=2)  # forces same-pool eviction
    assert pool.get_allocator_stats()["cross_pool_eviction_count"] == 0.0


def test_stats_round_trip_through_schema_1_1_0(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    _drive_mixed_workload(pool)

    metrics = AllocatorMetrics()
    metrics.allocator_specific_stats = dict(pool.get_allocator_stats())
    run = BenchmarkRun(
        spec=PRESETS["uniform_short"],
        allocator_name="avmp_static",
        hardware=Hardware(
            device="cpu",
            gpu_name=None,
            vram_total_bytes=None,
            cuda_capability=None,
        ),
        environment=Environment(
            torch_version="x",
            numpy_version="x",
            cachepawl_version="x",
            cuda_version=None,
            python_version="x",
        ),
        started_at="2026-05-15T00:00:00Z",
        finished_at="2026-05-15T00:00:01Z",
        metrics=metrics,
    )
    reloaded = BenchmarkRun.from_json(run.to_json())
    assert reloaded.metrics.allocator_specific_stats == metrics.allocator_specific_stats
    assert reloaded.metrics.allocator_specific_stats["mamba_ratio"] == 0.5
    assert reloaded.metrics.allocator_specific_stats["cross_pool_eviction_count"] == 0.0
    # v2 sub-PR 1 (RFC 0002 section 4.7) keys flow through the open-shape
    # allocator_specific_stats dict without a schema bump.
    assert reloaded.metrics.allocator_specific_stats["current_pressure_state_code"] == 0.0
    assert reloaded.metrics.allocator_specific_stats["rebalance_enabled"] == 0.0
