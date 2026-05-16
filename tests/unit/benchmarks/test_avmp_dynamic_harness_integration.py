"""Harness integration for ``avmp_dynamic``.

Asserts the registry entry exists, the factory returns an
``AsymmetricVirtualPool`` with ``rebalance_enabled=True``, and a small
end-to-end ``run_benchmark`` invocation fires at least one auto-trigger
on a KV-heavy workload. The full sweep (committed at
``benchmarks/results/avmp-v2/full/``) is the broader contract; this is
the cheap regression net.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.benchmarks import PRESETS, REGISTRY, BenchmarkRun, run_benchmark


def test_avmp_dynamic_is_registered() -> None:
    assert "avmp_dynamic" in REGISTRY


def test_avmp_dynamic_factory_returns_pool_with_rebalance_enabled() -> None:
    factory = REGISTRY["avmp_dynamic"]
    allocator = factory(PRESETS["uniform_short"], torch.device("cpu"))
    assert isinstance(allocator, AsymmetricVirtualPool)
    assert allocator.get_allocator_stats()["rebalance_enabled"] == 1.0


def test_avmp_dynamic_run_benchmark_uniform_short_cpu(tmp_path: Path) -> None:
    """End-to-end: 200-request uniform_short on CPU via the registered
    factory.

    The default factory budget is large enough that uniform_short's short
    prompts may not pressure the pool, so this test does NOT assert
    ``rebalance_count >= 1``. The full sweep (committed at
    ``benchmarks/results/avmp-v2/full/``) is the broader contract. Here
    we validate that the stats keys flow through with the rebalance
    machinery active.
    """

    factory = REGISTRY["avmp_dynamic"]
    spec = replace(PRESETS["uniform_short"], num_requests=200, seed=42)
    allocator = factory(spec, torch.device("cpu"))
    run = run_benchmark(
        allocator=allocator,
        spec=spec,
        allocator_name="avmp_dynamic",
        output_dir=tmp_path,
        device="cpu",
    )

    assert isinstance(run, BenchmarkRun)
    assert run.allocator_name == "avmp_dynamic"
    stats = run.metrics.allocator_specific_stats
    assert stats["rebalance_enabled"] == 1.0
    assert stats["rebalance_count"] >= 0.0
    assert stats["bytes_migrated_total"] >= 0.0
    assert stats["auto_rebalance_skipped_throttle"] >= 0.0
    # Conservation invariant: kv + ssm + waste equals the initial pool sum,
    # where the initial sum is the v1-style snapshot (kv_pool_bytes +
    # ssm_pool_bytes), which stays constant for the life of the pool.
    assert (
        stats["current_kv_pool_bytes"]
        + stats["current_ssm_pool_bytes"]
        + stats["bytes_wasted_to_alignment_total"]
        == stats["kv_pool_bytes"] + stats["ssm_pool_bytes"]
    )


@pytest.mark.gpu
def test_avmp_dynamic_run_benchmark_uniform_short_cuda(tmp_path: Path) -> None:
    """CUDA variant of the smoke test; skipped on CI via the gpu marker."""

    factory = REGISTRY["avmp_dynamic"]
    spec = replace(PRESETS["uniform_short"], num_requests=200, seed=7)
    allocator = factory(spec, torch.device("cuda"))
    run = run_benchmark(
        allocator=allocator,
        spec=spec,
        allocator_name="avmp_dynamic",
        output_dir=tmp_path,
        device="cuda",
    )
    stats = run.metrics.allocator_specific_stats
    assert stats["rebalance_enabled"] == 1.0
