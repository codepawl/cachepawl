"""Schema 1.0.0 to 1.1.0 backward-compat tests."""

from __future__ import annotations

import json

import pytest

from cachepawl.benchmarks import (
    SCHEMA_VERSION,
    AllocatorMetrics,
    BenchmarkRun,
)


def _v1_0_0_payload() -> dict[str, object]:
    """JSON shape that a 1.0.0 BenchmarkRun would have written.

    Mirrors the structure produced by the harness PR's smoke test
    fixture; the only difference from a 1.1.0 payload is the missing
    ``allocator_specific_stats`` key inside ``metrics``.
    """

    return {
        "schema_version": "1.0.0",
        "spec": {
            "name": "uniform_short",
            "num_requests": 8,
            "attention_layers": 4,
            "ssm_layers": 28,
            "attention_profile": {"num_kv_heads": 8, "head_dim": 128},
            "ssm_profile": {"d_inner": 8192, "d_state": 16},
            "dtype": "bf16",
            "seed": 1,
        },
        "allocator_name": "legacy",
        "hardware": {
            "device": "cpu",
            "gpu_name": None,
            "vram_total_bytes": None,
            "cuda_capability": None,
        },
        "environment": {
            "torch_version": "2.12.0+cpu",
            "numpy_version": "2.2.6",
            "cachepawl_version": "0.1.0",
            "cuda_version": None,
            "python_version": "3.10.19",
        },
        "started_at": "2026-05-01T00:00:00Z",
        "finished_at": "2026-05-01T00:00:01Z",
        "metrics": {
            "peak_reserved_bytes": 4096,
            "peak_allocated_bytes": 2048,
            "fragmentation_samples": [0.5, 0.25],
            "allocate_latency_ns": [100, 200],
            "free_latency_ns": [50],
            "allocate_latency_percentiles": {
                "p50_ns": 150,
                "p95_ns": 200,
                "p99_ns": 200,
                "max_ns": 200,
            },
            "free_latency_percentiles": {"p50_ns": 50, "p95_ns": 50, "p99_ns": 50, "max_ns": 50},
            "oom_count": 0,
            "preemption_count": 0,
            "active_requests_samples": [0, 1, 1, 0],
        },
        "notes": "legacy 1.0.0 artifact",
    }


def test_legacy_1_0_0_json_deserializes_with_empty_allocator_specific_stats() -> None:
    payload = _v1_0_0_payload()
    decoded = BenchmarkRun.from_json(json.dumps(payload))
    assert decoded.schema_version == "1.0.0"
    assert decoded.allocator_name == "legacy"
    assert decoded.metrics.allocator_specific_stats == {}


def test_1_1_0_round_trip_preserves_mapping() -> None:
    payload = _v1_0_0_payload()
    payload["schema_version"] = "1.1.0"
    metrics_dict = payload["metrics"]
    assert isinstance(metrics_dict, dict)
    metrics_dict["allocator_specific_stats"] = {
        "padding_waste_bytes": 1024.0,
        "num_pages_total": 64.0,
        "num_pages_used": 16.0,
    }
    decoded = BenchmarkRun.from_json(json.dumps(payload))
    assert decoded.metrics.allocator_specific_stats == {
        "padding_waste_bytes": 1024.0,
        "num_pages_total": 64.0,
        "num_pages_used": 16.0,
    }
    reencoded = BenchmarkRun.from_json(decoded.to_json())
    assert reencoded.metrics.allocator_specific_stats == decoded.metrics.allocator_specific_stats


def test_unknown_major_version_is_rejected() -> None:
    payload = _v1_0_0_payload()
    payload["schema_version"] = "2.0.0"
    with pytest.raises(ValueError, match="unsupported schema_version"):
        BenchmarkRun.from_json(json.dumps(payload))


def test_schema_version_constant_is_1_1_0() -> None:
    assert SCHEMA_VERSION == "1.1.0"


def test_allocator_metrics_default_factory_yields_empty_mapping() -> None:
    metrics = AllocatorMetrics()
    assert metrics.allocator_specific_stats == {}
    metrics.allocator_specific_stats["padding_waste_bytes"] = 42.0
    assert metrics.allocator_specific_stats == {"padding_waste_bytes": 42.0}
