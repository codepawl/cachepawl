"""Tests for planner comparison records and CLI output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cachepawl.bench.environment import RuntimeEnvironment
from cachepawl.bench.planner_comparison import (
    compare_planners,
    render_csv_summary,
    render_jsonl,
    render_markdown_summary,
)
from cachepawl.bench.result_schema import CacheProbeResult, GpuMetadata


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


def test_compare_planners_emits_one_record_per_backend_and_workload() -> None:
    records = compare_planners(
        workloads=("short-heavy", "mixed"),
        seed=1,
        num_requests=4,
        environment=_environment(),
    )
    assert len(records) == 4
    assert {(r.backend, r.workload) for r in records} == {
        ("vllm-style-padded", "short-heavy"),
        ("cachepawl-avmp", "short-heavy"),
        ("vllm-style-padded", "mixed"),
        ("cachepawl-avmp", "mixed"),
    }


def test_comparison_records_round_trip_through_schema() -> None:
    records = compare_planners(
        workloads=("mixed",),
        seed=1,
        num_requests=8,
        environment=_environment(),
    )
    lines = render_jsonl(records).splitlines()
    decoded = tuple(CacheProbeResult.from_json_line(line) for line in lines)
    assert decoded == records


def test_summary_renderers_include_paper_table_fields() -> None:
    records = compare_planners(
        workloads=("mixed",),
        seed=1,
        num_requests=8,
        environment=_environment(),
    )
    markdown = render_markdown_summary(records)
    csv = render_csv_summary(records)
    for text in (markdown, csv):
        assert "useful_bytes" in text
        assert "estimated_bytes" in text
        assert "overestimation_ratio" in text
        assert "wasted_fraction" in text
        assert "virtual_oom" in text
        assert "planner_runtime_us" in text


def test_comparison_uses_correct_metric_semantics() -> None:
    records = compare_planners(
        workloads=("mixed",),
        seed=1,
        num_requests=8,
        environment=_environment(),
    )
    padded = next(record for record in records if record.backend == "vllm-style-padded")
    avmp = next(record for record in records if record.backend == "cachepawl-avmp")
    assert padded.estimated_bytes == 614_465_536
    assert padded.useful_bytes == 196_526_080
    assert padded.overestimation_ratio == pytest.approx(3.1266, rel=1e-4)
    assert padded.wasted_fraction == pytest.approx(0.6802, rel=1e-4)
    assert avmp.overestimation_ratio == pytest.approx(1.0, rel=0.01)


def test_cli_is_deterministic_and_writes_jsonl_and_summary(tmp_path: Path) -> None:
    first_jsonl = tmp_path / "first.jsonl"
    second_jsonl = tmp_path / "second.jsonl"
    first_summary = tmp_path / "first.md"
    second_summary = tmp_path / "second.md"
    base_cmd = [
        sys.executable,
        "benchmarks/scripts/compare_cache_planners.py",
        "--workload",
        "mixed",
        "--seed",
        "1",
        "--num-requests",
        "8",
        "--gpu-total-bytes",
        str(12 * 1024**3),
    ]
    subprocess.run(
        [*base_cmd, "--jsonl-output", str(first_jsonl), "--summary-output", str(first_summary)],
        check=True,
    )
    subprocess.run(
        [*base_cmd, "--jsonl-output", str(second_jsonl), "--summary-output", str(second_summary)],
        check=True,
    )
    assert first_jsonl.read_text() == second_jsonl.read_text()
    assert first_summary.read_text() == second_summary.read_text()
    decoded = [
        CacheProbeResult.from_json_line(line) for line in first_jsonl.read_text().splitlines()
    ]
    assert {record.backend for record in decoded} == {"vllm-style-padded", "cachepawl-avmp"}


def test_artifact_pack_cli_writes_expected_reference_files(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    base_cmd = [
        sys.executable,
        "benchmarks/scripts/create_planner_comparison_pack.py",
        "--seed",
        "1",
        "--num-requests",
        "8",
        "--gpu-name",
        "NVIDIA GeForce RTX 3060",
        "--gpu-total-bytes",
        str(12 * 1024**3),
    ]
    subprocess.run([*base_cmd, "--output-dir", str(first)], check=True)
    subprocess.run([*base_cmd, "--output-dir", str(second)], check=True)

    expected_files = {
        "README.md",
        "environment.json",
        "manifest.json",
        "summary.md",
        "short-heavy.jsonl",
        "long-heavy.jsonl",
        "mixed.jsonl",
    }
    assert {path.name for path in first.iterdir()} == expected_files
    for name in expected_files:
        assert (first / name).read_text() == (second / name).read_text()

    summary = (first / "summary.md").read_text()
    assert "overestimation_ratio" in summary
    assert "wasted_fraction" in summary
    assert "vllm-style-padded" in summary
    assert "cachepawl-avmp" in summary
    readme = (first / "README.md").read_text()
    assert "AVMP can reduce overestimation" in readme
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest == {
        "artifact_name": "rtx3060-planner-comparison",
        "generated_at": "1970-01-01T00:00:00Z",
        "seed": 1,
        "num_requests": 8,
        "workloads": ["short-heavy", "long-heavy", "mixed"],
        "backends": ["vllm-style-padded", "cachepawl-avmp"],
        "target_gpu_name": "NVIDIA GeForce RTX 3060",
        "target_gpu_total_bytes": 12 * 1024**3,
        "runtime_measurement_enabled": False,
        "schema_version": "0.1.0",
        "generation_command": (
            "UV_CACHE_DIR=/tmp/uv-cache uv run python "
            "benchmarks/scripts/create_planner_comparison_pack.py "
            "--output-dir benchmarks/results/rtx3060/planner-comparison "
            "--seed 1 --num-requests 8 --gpu-name "
            '"NVIDIA GeForce RTX 3060" --gpu-total-bytes 12884901888'
        ),
    }
    for workload in ("short-heavy", "long-heavy", "mixed"):
        decoded = [
            CacheProbeResult.from_json_line(line)
            for line in (first / f"{workload}.jsonl").read_text().splitlines()
        ]
        assert len(decoded) == 2
        assert {record.workload for record in decoded} == {workload}
        assert {record.backend for record in decoded} == {
            "vllm-style-padded",
            "cachepawl-avmp",
        }
