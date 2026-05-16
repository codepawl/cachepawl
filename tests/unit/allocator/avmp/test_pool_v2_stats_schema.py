"""Round-trip the v2 sub-PR 1 stats through schema 1.1.0.

The 12 new keys land in the open-shape ``allocator_specific_stats`` dict
(``dict[str, float]``), so no schema version bump is required. This test
exercises the full BenchmarkRun.to_json / from_json cycle and asserts every
new key survives with ``float`` type and the documented default.
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
from cachepawl.models.spec import HybridModelSpec

_NEW_V2_KEYS = (
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
)


def _wrap(stats: dict[str, float]) -> BenchmarkRun:
    metrics = AllocatorMetrics()
    metrics.allocator_specific_stats = stats
    return BenchmarkRun(
        spec=PRESETS["uniform_short"],
        allocator_name="avmp_static",
        hardware=Hardware(device="cpu", gpu_name=None, vram_total_bytes=None, cuda_capability=None),
        environment=Environment(
            torch_version="x",
            numpy_version="x",
            cachepawl_version="x",
            cuda_version=None,
            python_version="x",
        ),
        started_at="2026-05-16T00:00:00Z",
        finished_at="2026-05-16T00:00:01Z",
        metrics=metrics,
    )


def test_v2_keys_round_trip_through_schema_1_1_0(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = AsymmetricVirtualPool(
        model_spec=jamba_spec,
        total_bytes=4 * 1024 * 1024,
        device=cpu_device,
        mamba_ratio=0.5,
        rebalance_enabled=True,
    )
    stats = dict(pool.get_allocator_stats())

    reloaded = BenchmarkRun.from_json(_wrap(stats).to_json())
    round_tripped = reloaded.metrics.allocator_specific_stats

    for key in _NEW_V2_KEYS:
        assert key in round_tripped, f"key {key!r} dropped on round-trip"
        assert isinstance(round_tripped[key], float), (
            f"key {key!r} round-tripped as {type(round_tripped[key]).__name__}"
        )
    # v1 keys still survive (no regression on the existing contract).
    assert round_tripped["mamba_ratio"] == 0.5
    assert round_tripped["cross_pool_eviction_count"] == 0.0


def test_v2_defaults_match_rfc_0002(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Defaults visible in stats match RFC 0002 section 4.2 / section 5."""

    pool = AsymmetricVirtualPool(
        model_spec=jamba_spec,
        total_bytes=4 * 1024 * 1024,
        device=cpu_device,
        mamba_ratio=0.5,
        rebalance_enabled=True,
    )
    stats = pool.get_allocator_stats()
    assert stats["rebalance_enabled"] == 1.0
    assert stats["threshold_low"] == 0.05
    assert stats["threshold_high"] == 0.30
    assert stats["migration_batch_size"] == 1.0
