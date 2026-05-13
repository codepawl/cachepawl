"""Tests for the BenchmarkRun JSON schema round-trip."""

from __future__ import annotations

import json

from cachepawl.benchmarks import (
    PRESETS,
    SCHEMA_VERSION,
    AllocatorMetrics,
    BenchmarkRun,
    Environment,
    Hardware,
)
from cachepawl.quant.dtypes import DType


def _sample_run(allocator_name: str = "mock") -> BenchmarkRun:
    metrics = AllocatorMetrics(
        peak_reserved_bytes=4096,
        peak_allocated_bytes=2048,
        fragmentation_samples=[0.5, 0.25, 0.0],
        allocate_latency_ns=[100, 200, 300],
        free_latency_ns=[10, 20],
        oom_count=1,
        preemption_count=0,
        active_requests_samples=[0, 1, 2, 1],
    )
    return BenchmarkRun(
        spec=PRESETS["uniform_short"],
        allocator_name=allocator_name,
        hardware=Hardware(
            device="cpu",
            gpu_name=None,
            vram_total_bytes=None,
            cuda_capability=None,
        ),
        environment=Environment(
            torch_version="2.12.0+cpu",
            numpy_version="2.2.6",
            cachepawl_version="0.1.0",
            cuda_version=None,
            python_version="3.10.19",
        ),
        started_at="2026-05-13T12:00:00Z",
        finished_at="2026-05-13T12:00:01Z",
        metrics=metrics,
        notes="smoke",
    )


def test_schema_version_is_one_zero_zero() -> None:
    assert SCHEMA_VERSION == "1.1.0"
    assert _sample_run().schema_version == "1.1.0"


def test_to_json_and_from_json_round_trip() -> None:
    run = _sample_run()
    encoded = run.to_json()
    decoded = BenchmarkRun.from_json(encoded)
    assert decoded.allocator_name == run.allocator_name
    assert decoded.spec == run.spec
    assert decoded.hardware == run.hardware
    assert decoded.environment == run.environment
    assert decoded.started_at == run.started_at
    assert decoded.finished_at == run.finished_at
    assert decoded.notes == run.notes
    assert decoded.schema_version == run.schema_version
    assert decoded.metrics.peak_reserved_bytes == run.metrics.peak_reserved_bytes
    assert decoded.metrics.fragmentation_samples == run.metrics.fragmentation_samples
    assert decoded.metrics.allocate_latency_ns == run.metrics.allocate_latency_ns


def test_dtype_serializes_as_string() -> None:
    run = _sample_run()
    raw = json.loads(run.to_json())
    assert raw["spec"]["dtype"] == DType.BF16.value


def test_hardware_with_cuda_capability_round_trips() -> None:
    run = _sample_run()
    run.hardware = Hardware(
        device="cuda",
        gpu_name="NVIDIA GeForce RTX 3060",
        vram_total_bytes=12 * 1024**3,
        cuda_capability=(8, 6),
    )
    decoded = BenchmarkRun.from_json(run.to_json())
    assert decoded.hardware.cuda_capability == (8, 6)
    assert decoded.hardware.gpu_name == "NVIDIA GeForce RTX 3060"


def test_metrics_dict_includes_percentile_summary() -> None:
    run = _sample_run()
    raw = json.loads(run.to_json())
    assert "allocate_latency_percentiles" in raw["metrics"]
    assert "free_latency_percentiles" in raw["metrics"]
    assert raw["metrics"]["allocate_latency_percentiles"]["max_ns"] == 300
