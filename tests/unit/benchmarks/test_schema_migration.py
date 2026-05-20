"""Schema 1.0.0 / 1.1.0 / 1.2.0 / 1.3.0 backward-compat tests."""

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


def test_schema_version_constant_is_1_3_0() -> None:
    assert SCHEMA_VERSION == "1.3.0"


def test_allocator_metrics_default_factory_yields_empty_mapping() -> None:
    metrics = AllocatorMetrics()
    assert metrics.allocator_specific_stats == {}
    metrics.allocator_specific_stats["padding_waste_bytes"] = 42.0
    assert metrics.allocator_specific_stats == {"padding_waste_bytes": 42.0}


def test_load_1_1_0_json_defaults_throughput_fields() -> None:
    """A 1.1.0 artifact must deserialize with new 1.2.0 fields defaulted.

    The strict completion semantic and the wall-clock-derived
    metrics did not exist at 1.1.0, so loading an older JSON yields
    a metrics record with zeros for the six new floats and ``None``
    for ``time_to_first_oom_seconds``. This is the contract that
    keeps committed reference artifacts under
    ``benchmarks/results/`` loadable across schema bumps.
    """

    payload = _v1_0_0_payload()
    payload["schema_version"] = "1.1.0"
    metrics_dict = payload["metrics"]
    assert isinstance(metrics_dict, dict)
    metrics_dict["allocator_specific_stats"] = {"padding_waste_bytes": 1.0}

    decoded = BenchmarkRun.from_json(json.dumps(payload))

    assert decoded.schema_version == "1.1.0"
    assert decoded.metrics.effective_batch_size_mean == 0.0
    assert decoded.metrics.effective_batch_size_p50 == 0.0
    assert decoded.metrics.effective_batch_size_p95 == 0.0
    assert decoded.metrics.effective_batch_size_p99 == 0.0
    assert decoded.metrics.goodput_requests_per_second == 0.0
    assert decoded.metrics.completion_ratio == 0.0
    assert decoded.metrics.time_to_first_oom_seconds is None


def test_round_trip_1_2_0_preserves_throughput_fields() -> None:
    payload = _v1_0_0_payload()
    payload["schema_version"] = "1.2.0"
    metrics_dict = payload["metrics"]
    assert isinstance(metrics_dict, dict)
    metrics_dict["allocator_specific_stats"] = {}
    metrics_dict["effective_batch_size_mean"] = 4.5
    metrics_dict["effective_batch_size_p50"] = 4.0
    metrics_dict["effective_batch_size_p95"] = 7.0
    metrics_dict["effective_batch_size_p99"] = 8.0
    metrics_dict["goodput_requests_per_second"] = 50.0
    metrics_dict["completion_ratio"] = 0.95
    metrics_dict["time_to_first_oom_seconds"] = 1.25

    decoded = BenchmarkRun.from_json(json.dumps(payload))
    reencoded = BenchmarkRun.from_json(decoded.to_json())

    assert reencoded.metrics.effective_batch_size_mean == 4.5
    assert reencoded.metrics.effective_batch_size_p50 == 4.0
    assert reencoded.metrics.effective_batch_size_p95 == 7.0
    assert reencoded.metrics.effective_batch_size_p99 == 8.0
    assert reencoded.metrics.goodput_requests_per_second == 50.0
    assert reencoded.metrics.completion_ratio == 0.95
    assert reencoded.metrics.time_to_first_oom_seconds == 1.25


def test_1_2_0_load_with_null_time_to_first_oom() -> None:
    payload = _v1_0_0_payload()
    payload["schema_version"] = "1.2.0"
    metrics_dict = payload["metrics"]
    assert isinstance(metrics_dict, dict)
    metrics_dict["allocator_specific_stats"] = {}
    metrics_dict["effective_batch_size_mean"] = 1.0
    metrics_dict["effective_batch_size_p50"] = 1.0
    metrics_dict["effective_batch_size_p95"] = 1.0
    metrics_dict["effective_batch_size_p99"] = 1.0
    metrics_dict["goodput_requests_per_second"] = 10.0
    metrics_dict["completion_ratio"] = 1.0
    metrics_dict["time_to_first_oom_seconds"] = None

    decoded = BenchmarkRun.from_json(json.dumps(payload))

    assert decoded.metrics.time_to_first_oom_seconds is None


def test_load_1_2_0_json_defaults_phase_time_fields() -> None:
    """A 1.2.0 artifact must deserialize with new 1.3.0 phase-time fields defaulted.

    Phase-time decomposition (service / oom_retry / migration / idle)
    landed in 1.3.0. Older committed sweep artifacts under
    ``benchmarks/results/`` predate it and must load with the four new
    int fields defaulting to 0 so the harness can still consume them.
    """

    payload = _v1_0_0_payload()
    payload["schema_version"] = "1.2.0"
    metrics_dict = payload["metrics"]
    assert isinstance(metrics_dict, dict)
    metrics_dict["allocator_specific_stats"] = {"time_spent_rebalancing_ns": 12345.0}
    metrics_dict["effective_batch_size_mean"] = 1.0
    metrics_dict["effective_batch_size_p50"] = 1.0
    metrics_dict["effective_batch_size_p95"] = 1.0
    metrics_dict["effective_batch_size_p99"] = 1.0
    metrics_dict["goodput_requests_per_second"] = 10.0
    metrics_dict["completion_ratio"] = 1.0
    metrics_dict["time_to_first_oom_seconds"] = 0.5

    decoded = BenchmarkRun.from_json(json.dumps(payload))

    assert decoded.schema_version == "1.2.0"
    assert decoded.metrics.time_in_service_ns == 0
    assert decoded.metrics.time_in_oom_retry_ns == 0
    assert decoded.metrics.time_in_migration_ns == 0
    assert decoded.metrics.time_in_idle_ns == 0


def test_round_trip_1_3_0_preserves_phase_time_fields() -> None:
    payload = _v1_0_0_payload()
    payload["schema_version"] = "1.3.0"
    metrics_dict = payload["metrics"]
    assert isinstance(metrics_dict, dict)
    metrics_dict["allocator_specific_stats"] = {}
    metrics_dict["effective_batch_size_mean"] = 4.5
    metrics_dict["effective_batch_size_p50"] = 4.0
    metrics_dict["effective_batch_size_p95"] = 7.0
    metrics_dict["effective_batch_size_p99"] = 8.0
    metrics_dict["goodput_requests_per_second"] = 50.0
    metrics_dict["completion_ratio"] = 0.95
    metrics_dict["time_to_first_oom_seconds"] = 1.25
    metrics_dict["time_in_service_ns"] = 1_500_000
    metrics_dict["time_in_oom_retry_ns"] = 750_000
    metrics_dict["time_in_migration_ns"] = 250_000
    metrics_dict["time_in_idle_ns"] = 9_000_000

    decoded = BenchmarkRun.from_json(json.dumps(payload))
    reencoded = BenchmarkRun.from_json(decoded.to_json())

    assert reencoded.schema_version == "1.3.0"
    assert reencoded.metrics.time_in_service_ns == 1_500_000
    assert reencoded.metrics.time_in_oom_retry_ns == 750_000
    assert reencoded.metrics.time_in_migration_ns == 250_000
    assert reencoded.metrics.time_in_idle_ns == 9_000_000


def test_allocator_metrics_phase_time_default_to_zero() -> None:
    metrics = AllocatorMetrics()
    assert metrics.time_in_service_ns == 0
    assert metrics.time_in_oom_retry_ns == 0
    assert metrics.time_in_migration_ns == 0
    assert metrics.time_in_idle_ns == 0
