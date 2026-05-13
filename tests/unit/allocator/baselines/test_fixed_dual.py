"""Unit tests for FixedDualPool.

The pool faithfully reproduces SGLang's static dual-pool partition. The
tests assert two pathology-defining behaviors:

- KV exhaustion does **not** evict SSM (cross-pool isolation).
- ``pool_underused_bytes`` grows positively when the workload skews
  toward one pool while the other has slack (rigidity surfacing).
"""

from __future__ import annotations

import math

import pytest
import torch

from cachepawl.allocator.baselines import FixedDualPool, align_up
from cachepawl.allocator.policy import EvictionPolicy
from cachepawl.benchmarks import (
    PRESETS,
    AllocatorMetrics,
    BenchmarkRun,
    Environment,
    Hardware,
)
from cachepawl.models.spec import HybridModelSpec, LayerKind
from cachepawl.quant.dtypes import bytes_per_element


def _kv_page_size(spec: HybridModelSpec, attention_page_tokens: int = 16) -> int:
    elem = bytes_per_element(spec.dtype)
    raw = (
        int(2.0 * spec.attention_profile.num_kv_heads * spec.attention_profile.head_dim * elem)
        * attention_page_tokens
    )
    return align_up(raw)


def _ssm_block_size(spec: HybridModelSpec) -> int:
    elem = bytes_per_element(spec.dtype)
    raw = int(spec.ssm_profile.d_inner * spec.ssm_profile.d_state * elem)
    return align_up(raw)


def test_mamba_ratio_zero_is_rejected(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    with pytest.raises(ValueError, match="mamba_ratio"):
        FixedDualPool(
            model_spec=jamba_spec,
            total_bytes=4 * 1024**2,
            device=cpu_device,
            mamba_ratio=0.0,
        )


def test_mamba_ratio_one_is_rejected(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    with pytest.raises(ValueError, match="mamba_ratio"):
        FixedDualPool(
            model_spec=jamba_spec,
            total_bytes=4 * 1024**2,
            device=cpu_device,
            mamba_ratio=1.0,
        )


def test_unsupported_eviction_policy_raises(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    with pytest.raises(NotImplementedError, match="LRU"):
        FixedDualPool(
            model_spec=jamba_spec,
            total_bytes=4 * 1024**2,
            device=cpu_device,
            eviction=EvictionPolicy.FIFO,
        )


def test_kv_exhaustion_does_not_evict_ssm(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    total_bytes = 4 * 1024**2
    pool = FixedDualPool(
        model_spec=jamba_spec,
        total_bytes=total_bytes,
        device=cpu_device,
        mamba_ratio=0.5,
    )
    stats_before = pool.get_allocator_stats()
    ssm_blocks_total = int(stats_before["ssm_blocks_total"])
    kv_pages_total = int(stats_before["kv_pages_total"])

    pool.set_current_layer_kind(LayerKind.MAMBA2)
    for i in range(ssm_blocks_total):
        pool.set_current_request_id(100 + i)
        pool.allocate(1, dtype_bytes=2)
    assert int(pool.get_allocator_stats()["ssm_blocks_used"]) == ssm_blocks_total

    pool.set_current_layer_kind(LayerKind.ATTENTION)
    for i in range(kv_pages_total):
        pool.set_current_request_id(200 + i)
        pool.allocate(1, dtype_bytes=2)
    assert int(pool.get_allocator_stats()["kv_pages_used"]) == kv_pages_total

    pool.set_current_request_id(300)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="fixed_dual"):
        pool.allocate(kv_pages_total + 1, dtype_bytes=2)

    final_stats = pool.get_allocator_stats()
    assert int(final_stats["ssm_blocks_used"]) == ssm_blocks_total, (
        "SSM pool must remain untouched when KV exhaustion triggers OOM"
    )


def test_pool_underused_bytes_under_skewed_workload(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    total_bytes = 4 * 1024**2
    pool = FixedDualPool(
        model_spec=jamba_spec,
        total_bytes=total_bytes,
        device=cpu_device,
        mamba_ratio=0.5,
    )
    stats0 = pool.get_allocator_stats()
    ssm_blocks_total = int(stats0["ssm_blocks_total"])

    pool.set_current_layer_kind(LayerKind.MAMBA2)
    for i in range(ssm_blocks_total):
        pool.set_current_request_id(i)
        pool.allocate(1, dtype_bytes=2)

    pool.set_current_request_id(9999)
    pool.allocate(1, dtype_bytes=2)

    stats_after = pool.get_allocator_stats()
    assert stats_after["pool_underused_bytes_ssm"] > 0.0
    assert stats_after["pool_underused_bytes_kv"] == 0.0


def test_stats_round_trip_schema_1_1_0(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = FixedDualPool(
        model_spec=jamba_spec,
        total_bytes=4 * 1024**2,
        device=cpu_device,
        mamba_ratio=0.5,
    )
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(0)
    pool.allocate(1, dtype_bytes=2)
    pool.set_current_layer_kind(LayerKind.MAMBA2)
    pool.allocate(1, dtype_bytes=2)

    metrics = AllocatorMetrics()
    metrics.allocator_specific_stats = dict(pool.get_allocator_stats())
    run = BenchmarkRun(
        spec=PRESETS["uniform_short"],
        allocator_name="fixed_dual",
        hardware=Hardware(device="cpu", gpu_name=None, vram_total_bytes=None, cuda_capability=None),
        environment=Environment(
            torch_version="x",
            numpy_version="x",
            cachepawl_version="x",
            cuda_version=None,
            python_version="x",
        ),
        started_at="2026-05-13T00:00:00Z",
        finished_at="2026-05-13T00:00:01Z",
        metrics=metrics,
    )
    reloaded = BenchmarkRun.from_json(run.to_json())
    assert reloaded.metrics.allocator_specific_stats == metrics.allocator_specific_stats
    assert reloaded.metrics.allocator_specific_stats["mamba_ratio"] == 0.5


def test_page_size_math_matches_first_principles(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    total_bytes = 4 * 1024**2
    pool = FixedDualPool(
        model_spec=jamba_spec,
        total_bytes=total_bytes,
        device=cpu_device,
        mamba_ratio=0.5,
    )
    stats = pool.get_allocator_stats()
    kv_page = _kv_page_size(jamba_spec)
    ssm_block = _ssm_block_size(jamba_spec)
    expected_kv_pages = (total_bytes // 2) // kv_page
    expected_ssm_blocks = (total_bytes // 2) // ssm_block
    assert int(stats["kv_pages_total"]) == expected_kv_pages
    assert int(stats["ssm_blocks_total"]) == expected_ssm_blocks


@pytest.mark.gpu
def test_fixed_dual_uniform_short_on_cuda(tmp_path: object) -> None:
    from pathlib import Path

    from cachepawl.benchmarks import REGISTRY, run_benchmark

    factory = REGISTRY["fixed_dual"]
    spec = PRESETS["uniform_short"]
    short_spec = type(spec)(
        name=spec.name,
        num_requests=50,
        attention_layers=spec.attention_layers,
        ssm_layers=spec.ssm_layers,
        attention_profile=spec.attention_profile,
        ssm_profile=spec.ssm_profile,
        dtype=spec.dtype,
        seed=spec.seed,
    )
    device = torch.device("cuda")
    allocator = factory(short_spec, device)
    assert isinstance(tmp_path, Path)
    run = run_benchmark(
        allocator=allocator,
        spec=short_spec,
        allocator_name="fixed_dual",
        output_dir=tmp_path,
        device="cuda",
    )
    assert run.schema_version == "1.1.0"
    stats = run.metrics.allocator_specific_stats
    assert math.isfinite(stats["pool_underused_bytes_kv"])
    assert math.isfinite(stats["pool_underused_bytes_ssm"])
