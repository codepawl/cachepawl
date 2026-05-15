"""Harness integration tests for ``avmp_static``.

Asserts the registry surface (presence + correct factory return type)
and a small end-to-end ``run_benchmark`` invocation that hits the
allocator's allocate/free path. The AVMP-specific data sanity
invariants (the four documented in ``benchmarks/README.md``) are
checked against the live ``allocator_specific_stats`` after the run.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.benchmarks import PRESETS, REGISTRY, BenchmarkRun, run_benchmark

_EXPECTED_AVMP_KEYS: frozenset[str] = frozenset(
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

    # Four documented sanity invariants from benchmarks/README.md.
    assert stats["kv_pages_used"] + stats["kv_pages_free"] == stats["kv_pages_total"]
    assert stats["ssm_blocks_used"] + stats["ssm_blocks_free"] == stats["ssm_blocks_total"]
    assert stats["virtual_handles_live"] == stats["kv_pages_used"] + stats["ssm_blocks_used"]
    assert stats["cross_pool_eviction_count"] == 0.0


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
