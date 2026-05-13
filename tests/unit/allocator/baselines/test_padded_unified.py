"""Unit tests for PaddedUnifiedPool.

The pool faithfully reproduces vLLM's ``unify_kv_cache_spec_page_size``
behavior: attention pages and SSM blocks both pay the inflated page
cost. These tests verify the math by first principles, not via the
black-box stats, so a future regression that misreports the waste fails
loudly.
"""

from __future__ import annotations

import math

import pytest
import torch

from cachepawl.allocator.baselines import PaddedUnifiedPool, align_up
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


def _expected_page_size(spec: HybridModelSpec, attention_page_tokens: int = 16) -> int:
    elem = bytes_per_element(spec.dtype)
    attn_logical = (
        int(2.0 * spec.attention_profile.num_kv_heads * spec.attention_profile.head_dim * elem)
        * attention_page_tokens
    )
    ssm_logical = int(spec.ssm_profile.d_inner * spec.ssm_profile.d_state * elem)
    return align_up(max(attn_logical, ssm_logical))


def _attention_logical_bytes(spec: HybridModelSpec, attention_page_tokens: int = 16) -> int:
    elem = bytes_per_element(spec.dtype)
    return (
        int(2.0 * spec.attention_profile.num_kv_heads * spec.attention_profile.head_dim * elem)
        * attention_page_tokens
    )


def test_capacity_returns_after_allocate_then_free(
    synthetic_large_state_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    page_size = _expected_page_size(synthetic_large_state_spec)
    pool = PaddedUnifiedPool(
        model_spec=synthetic_large_state_spec,
        total_bytes=page_size * 8,
        device=cpu_device,
    )
    initial_free = pool.stats().free_blocks
    ids = pool.allocate(3, dtype_bytes=2)
    assert len(ids) == 3
    assert pool.stats().free_blocks == initial_free - 3
    pool.free(ids)
    assert pool.stats().free_blocks == initial_free


def test_padding_waste_math_matches_formula(
    synthetic_large_state_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    page_size = _expected_page_size(synthetic_large_state_spec)
    attn_logical = _attention_logical_bytes(synthetic_large_state_spec)
    expected_waste_per_page = page_size - attn_logical
    assert expected_waste_per_page > 0, (
        "synthetic spec must have SSM block size > attention page so waste is non-zero"
    )

    pool = PaddedUnifiedPool(
        model_spec=synthetic_large_state_spec,
        total_bytes=page_size * 4,
        device=cpu_device,
    )
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(0)
    pool.allocate(1, dtype_bytes=2)
    stats = pool.get_allocator_stats()
    assert stats["padding_waste_bytes"] == float(expected_waste_per_page)


def test_lru_eviction_drops_oldest_request(
    synthetic_large_state_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    page_size = _expected_page_size(synthetic_large_state_spec)
    pool = PaddedUnifiedPool(
        model_spec=synthetic_large_state_spec,
        total_bytes=page_size * 3,
        device=cpu_device,
    )
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(0)
    pool.allocate(1, dtype_bytes=2)
    pool.set_current_request_id(1)
    pool.allocate(1, dtype_bytes=2)
    pool.set_current_request_id(2)
    pool.allocate(1, dtype_bytes=2)
    assert pool.stats().free_blocks == 0

    pool.set_current_request_id(3)
    new_ids = pool.allocate(1, dtype_bytes=2)
    assert len(new_ids) == 1
    assert pool.stats().free_blocks == 0


def test_oom_when_no_evictable_request(
    synthetic_large_state_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    page_size = _expected_page_size(synthetic_large_state_spec)
    pool = PaddedUnifiedPool(
        model_spec=synthetic_large_state_spec,
        total_bytes=page_size,
        device=cpu_device,
    )
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(0)
    pool.allocate(1, dtype_bytes=2)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="padded_unified"):
        pool.allocate(2, dtype_bytes=2)


def test_unsupported_eviction_policy_raises(
    synthetic_large_state_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    with pytest.raises(NotImplementedError, match="LRU"):
        PaddedUnifiedPool(
            model_spec=synthetic_large_state_spec,
            total_bytes=_expected_page_size(synthetic_large_state_spec) * 4,
            device=cpu_device,
            eviction=EvictionPolicy.LFU,
        )


def test_stats_round_trip_schema_1_1_0(
    synthetic_large_state_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    page_size = _expected_page_size(synthetic_large_state_spec)
    pool = PaddedUnifiedPool(
        model_spec=synthetic_large_state_spec,
        total_bytes=page_size * 8,
        device=cpu_device,
    )
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(0)
    pool.allocate(2, dtype_bytes=2)

    metrics = AllocatorMetrics()
    metrics.allocator_specific_stats = dict(pool.get_allocator_stats())
    run = BenchmarkRun(
        spec=PRESETS["uniform_short"],
        allocator_name="padded_unified",
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
    assert reloaded.metrics.allocator_specific_stats["padding_waste_bytes"] > 0


@pytest.mark.gpu
def test_padded_unified_uniform_short_on_cuda(tmp_path: object) -> None:
    from pathlib import Path

    from cachepawl.benchmarks import REGISTRY, run_benchmark

    factory = REGISTRY["padded_unified"]
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
        allocator_name="padded_unified",
        output_dir=tmp_path,
        device="cuda",
    )
    assert run.schema_version == "1.1.0"
    assert math.isfinite(run.metrics.allocator_specific_stats["padding_waste_bytes"])
