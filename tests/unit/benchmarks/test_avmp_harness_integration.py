"""Harness integration tests for ``avmp_static``.

Asserts the registry surface (presence + correct factory return type)
and a small end-to-end ``run_benchmark`` invocation that hits the
allocator's allocate/free path. The AVMP-specific data sanity
invariants documented in ``benchmarks/README.md`` are checked against
the live ``allocator_specific_stats`` after the run.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.benchmarks import PRESETS, REGISTRY, BenchmarkRun, run_benchmark

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
# thirteenth, ``bytes_wasted_to_alignment_total``. The avmp_static factory
# constructs with rebalance_enabled=False AND calls no manual trigger, so
# the migration counters and pressure-state code stay at their defaults.
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

_EXPECTED_AVMP_KEYS: frozenset[str] = _EXPECTED_V1_KEYS | _EXPECTED_V2_KEYS


def test_avmp_static_is_registered() -> None:
    assert "avmp_static" in REGISTRY


def test_avmp_factory_returns_asymmetric_virtual_pool() -> None:
    factory = REGISTRY["avmp_static"]
    allocator = factory(PRESETS["uniform_short"], torch.device("cpu"))
    assert isinstance(allocator, AsymmetricVirtualPool)


def test_avmp_run_benchmark_uniform_short_cpu(tmp_path: Path) -> None:
    """End-to-end: a 50-request uniform_short run on CPU, no exceptions, valid stats."""

    factory = REGISTRY["avmp_static"]
    spec = replace(PRESETS["uniform_short"], num_requests=50, seed=42)
    allocator = factory(spec, torch.device("cpu"))
    run = run_benchmark(
        allocator=allocator,
        spec=spec,
        allocator_name="avmp_static",
        output_dir=tmp_path,
        device="cpu",
    )

    assert isinstance(run, BenchmarkRun)
    assert run.allocator_name == "avmp_static"
    stats = run.metrics.allocator_specific_stats
    assert frozenset(stats.keys()) == _EXPECTED_AVMP_KEYS

    # v1 sanity invariants from benchmarks/README.md.
    assert stats["kv_pages_used"] + stats["kv_pages_free"] == stats["kv_pages_total"]
    assert stats["ssm_blocks_used"] + stats["ssm_blocks_free"] == stats["ssm_blocks_total"]
    assert stats["virtual_handles_live"] == stats["kv_pages_used"] + stats["ssm_blocks_used"]
    assert stats["cross_pool_eviction_count"] == 0.0
    # v2 invariants: harness factory uses rebalance_enabled=False and never
    # calls trigger_manual_rebalance, so the monitor is absent and the
    # migration counters stay at 0.
    assert stats["rebalance_enabled"] == 0.0
    assert stats["rebalance_count"] == 0.0
    assert stats["bytes_migrated_total"] == 0.0
    assert stats["time_spent_rebalancing_ns"] == 0.0
    assert stats["bytes_wasted_to_alignment_total"] == 0.0
    assert stats["current_pressure_state_code"] == 0.0  # BALANCED
    assert 0.0 <= stats["kv_free_ratio"] <= 1.0
    assert 0.0 <= stats["ssm_free_ratio"] <= 1.0


@pytest.mark.gpu
def test_avmp_run_benchmark_uniform_short_cuda(tmp_path: Path) -> None:
    """200-request CUDA run; skipped on CI via the gpu marker."""

    factory = REGISTRY["avmp_static"]
    spec = replace(PRESETS["uniform_short"], num_requests=200, seed=7)
    allocator = factory(spec, torch.device("cuda"))
    run = run_benchmark(
        allocator=allocator,
        spec=spec,
        allocator_name="avmp_static",
        output_dir=tmp_path,
        device="cuda",
    )

    stats = run.metrics.allocator_specific_stats
    assert frozenset(stats.keys()) == _EXPECTED_AVMP_KEYS
    assert stats["cross_pool_eviction_count"] == 0.0
