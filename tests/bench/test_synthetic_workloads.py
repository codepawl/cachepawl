"""Tests for deterministic planner-only synthetic workloads."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cachepawl.bench.environment import RuntimeEnvironment
from cachepawl.bench.result_schema import CacheProbeResult, GpuMetadata
from cachepawl.bench.synthetic_workloads import (
    PLANNER_BACKENDS,
    SYNTHETIC_WORKLOADS,
    build_probe_result,
    generate_synthetic_workload,
)


def _environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        gpu=GpuMetadata(
            name="NVIDIA GeForce RTX 3060",
            total_memory_bytes=12 * 1024**3,
            compute_capability=(8, 6),
            cuda_available=False,
            device_count=0,
        ),
        metadata={"python_version": "test", "vllm_version": None},
    )


def test_workload_names_are_exactly_the_requested_set() -> None:
    assert set(SYNTHETIC_WORKLOADS) == {"short-heavy", "long-heavy", "mixed"}


def test_planner_backends_are_stable() -> None:
    assert set(PLANNER_BACKENDS) == {"padded-unified", "avmp-static", "fixed-dual"}


def test_fixed_seed_produces_identical_workload() -> None:
    first = generate_synthetic_workload("mixed", seed=123, num_requests=16)
    second = generate_synthetic_workload("mixed", seed=123, num_requests=16)
    assert first == second


def test_different_seed_changes_workload() -> None:
    first = generate_synthetic_workload("mixed", seed=123, num_requests=16)
    second = generate_synthetic_workload("mixed", seed=124, num_requests=16)
    assert first != second


def test_probe_result_contains_required_paper_fields() -> None:
    workload = generate_synthetic_workload("short-heavy", seed=1, num_requests=8)
    result = build_probe_result(
        backend="padded-unified",
        workload=workload,
        environment=_environment(),
        timestamp="1970-01-01T00:00:00Z",
    )
    raw = result.to_dict()
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
    assert result.reserved_bytes >= result.useful_bytes
    assert result.overestimation_ratio >= 1.0
    assert 0.0 <= result.wasted_fraction <= 1.0


def test_jsonl_is_deterministic_when_runtime_is_normalized() -> None:
    workload = generate_synthetic_workload("mixed", seed=7, num_requests=4)
    first = build_probe_result(
        backend="avmp-static",
        workload=workload,
        environment=_environment(),
        timestamp="1970-01-01T00:00:00Z",
    )
    second = build_probe_result(
        backend="avmp-static",
        workload=workload,
        environment=_environment(),
        timestamp="1970-01-01T00:00:00Z",
    )
    assert first.to_json_line() == second.to_json_line()


def test_cli_writes_jsonl(tmp_path: Path) -> None:
    out_path = tmp_path / "probe.jsonl"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/scripts/run_cache_probe.py",
            "--workload",
            "short-heavy",
            "--backend",
            "avmp-static",
            "--seed",
            "1",
            "--num-requests",
            "4",
            "--gpu-total-bytes",
            str(12 * 1024**3),
            "--output",
            str(out_path),
        ],
        check=True,
    )
    lines = out_path.read_text().splitlines()
    assert len(lines) == 1
    result = CacheProbeResult.from_json_line(lines[0])
    assert result.workload == "short-heavy"
    assert result.backend == "avmp-static"


def test_cli_jsonl_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    base_cmd = [
        sys.executable,
        "benchmarks/scripts/run_cache_probe.py",
        "--workload",
        "mixed",
        "--backend",
        "padded-unified",
        "--seed",
        "7",
        "--num-requests",
        "4",
        "--gpu-total-bytes",
        str(12 * 1024**3),
    ]
    subprocess.run([*base_cmd, "--output", str(first_path)], check=True)
    subprocess.run([*base_cmd, "--output", str(second_path)], check=True)
    assert first_path.read_text() == second_path.read_text()
