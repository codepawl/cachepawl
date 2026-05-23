"""Tests for cache probe result schema validation and JSONL round-trip."""

from __future__ import annotations

import dataclasses
import json

import pytest

from cachepawl.bench.result_schema import (
    BENCH_RESULT_SCHEMA_VERSION,
    CacheProbeResult,
    GpuMetadata,
)


def _gpu() -> GpuMetadata:
    return GpuMetadata(
        name="NVIDIA GeForce RTX 3060",
        total_memory_bytes=12 * 1024**3,
        compute_capability=(8, 6),
        cuda_available=False,
        device_count=0,
    )


def _result() -> CacheProbeResult:
    return CacheProbeResult(
        run_id="run-1",
        timestamp="1970-01-01T00:00:00Z",
        backend="avmp-static",
        workload="short-heavy",
        model="jamba-1.5-mini",
        gpu=_gpu(),
        estimated_bytes=100,
        reserved_bytes=100,
        useful_bytes=80,
        overestimation_ratio=1.25,
        wasted_fraction=0.2,
        virtual_oom=False,
        planner_runtime_us=1.5,
        metadata={"seed": 1, "vllm_version": None, "cuda_available": False},
    )


def test_schema_version_is_pinned() -> None:
    assert BENCH_RESULT_SCHEMA_VERSION == "0.1.0"
    assert _result().schema_version == "0.1.0"


def test_json_line_round_trip_preserves_required_fields() -> None:
    result = _result()
    decoded = CacheProbeResult.from_json_line(result.to_json_line())
    assert decoded == result
    raw = json.loads(result.to_json_line())
    for field in (
        "run_id",
        "timestamp",
        "backend",
        "workload",
        "model",
        "gpu",
        "estimated_bytes",
        "reserved_bytes",
        "useful_bytes",
        "overestimation_ratio",
        "wasted_fraction",
        "virtual_oom",
        "planner_runtime_us",
        "metadata",
    ):
        assert field in raw


def test_result_is_frozen_slots_record() -> None:
    result = _result()
    assert not hasattr(result, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.run_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize("field", ["estimated_bytes", "reserved_bytes", "useful_bytes"])
def test_negative_byte_fields_are_rejected(field: str) -> None:
    kwargs = _result().to_dict()
    kwargs.pop("schema_version")
    kwargs["gpu"] = _gpu()
    kwargs[field] = -1
    with pytest.raises(ValueError, match=field):
        CacheProbeResult(**kwargs)  # type: ignore[arg-type]


def test_invalid_overestimation_ratio_rejected() -> None:
    with pytest.raises(ValueError, match="overestimation_ratio"):
        CacheProbeResult(
            run_id="run-1",
            timestamp="1970-01-01T00:00:00Z",
            backend="avmp-static",
            workload="short-heavy",
            model="jamba-1.5-mini",
            gpu=_gpu(),
            estimated_bytes=1,
            reserved_bytes=1,
            useful_bytes=1,
            overestimation_ratio=-1.0,
            wasted_fraction=0.0,
            virtual_oom=False,
            planner_runtime_us=0.0,
        )


def test_invalid_wasted_fraction_rejected() -> None:
    with pytest.raises(ValueError, match="wasted_fraction"):
        CacheProbeResult(
            run_id="run-1",
            timestamp="1970-01-01T00:00:00Z",
            backend="avmp-static",
            workload="short-heavy",
            model="jamba-1.5-mini",
            gpu=_gpu(),
            estimated_bytes=1,
            reserved_bytes=1,
            useful_bytes=1,
            overestimation_ratio=1.0,
            wasted_fraction=1.1,
            virtual_oom=False,
            planner_runtime_us=0.0,
        )


def test_nonzero_estimate_requires_positive_useful_bytes() -> None:
    with pytest.raises(ValueError, match="useful_bytes"):
        CacheProbeResult(
            run_id="run-1",
            timestamp="1970-01-01T00:00:00Z",
            backend="avmp-static",
            workload="short-heavy",
            model="jamba-1.5-mini",
            gpu=_gpu(),
            estimated_bytes=1,
            reserved_bytes=1,
            useful_bytes=0,
            overestimation_ratio=0.0,
            wasted_fraction=1.0,
            virtual_oom=False,
            planner_runtime_us=0.0,
        )


def test_metric_semantics_are_ratio_and_fraction() -> None:
    result = CacheProbeResult(
        run_id="run-1",
        timestamp="1970-01-01T00:00:00Z",
        backend="vllm-style-padded",
        workload="mixed",
        model="jamba-1.5-mini",
        gpu=_gpu(),
        estimated_bytes=614_465_536,
        reserved_bytes=614_465_536,
        useful_bytes=196_526_080,
        overestimation_ratio=614_465_536 / 196_526_080,
        wasted_fraction=(614_465_536 - 196_526_080) / 614_465_536,
        virtual_oom=False,
        planner_runtime_us=0.0,
    )
    assert result.overestimation_ratio == pytest.approx(3.1266, rel=1e-4)
    assert result.wasted_fraction == pytest.approx(0.6802, rel=1e-4)


def test_missing_required_key_rejected() -> None:
    raw = _result().to_dict()
    raw.pop("backend")
    with pytest.raises(ValueError, match="backend"):
        CacheProbeResult.from_dict(raw)


def test_non_object_json_root_rejected() -> None:
    with pytest.raises(ValueError, match="root"):
        CacheProbeResult.from_json_line("[]")
