"""End-to-end smoke tests for the runner and CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from cachepawl.benchmarks import PRESETS, REGISTRY, BenchmarkRun, run_benchmark
from cachepawl.benchmarks.run import main as cli_main
from tests.unit.benchmarks.conftest import FakeAllocator


@pytest.fixture
def registered_mock_allocator() -> object:
    """Register a fresh FakeAllocator under the name 'mock' for one test."""

    REGISTRY["mock"] = FakeAllocator
    yield None
    REGISTRY.pop("mock", None)


def test_run_benchmark_writes_json_and_returns_run(tmp_path: Path) -> None:
    spec = replace(PRESETS["uniform_short"], num_requests=50, seed=42)
    allocator = FakeAllocator(total_blocks=1_000_000)
    run = run_benchmark(
        allocator=allocator,
        spec=spec,
        allocator_name="mock",
        output_dir=tmp_path,
        device="cpu",
    )

    assert isinstance(run, BenchmarkRun)
    assert run.allocator_name == "mock"
    assert run.spec.name == "uniform_short"
    assert run.metrics.allocate_latency_ns
    assert run.metrics.free_latency_ns
    assert run.metrics.fragmentation_samples
    assert allocator.allocate_calls > 0
    assert allocator.free_calls == spec.num_requests

    workload_dir = tmp_path / "mock" / "uniform_short"
    files = list(workload_dir.glob("*.json"))
    assert len(files) == 1
    reloaded = BenchmarkRun.from_json(files[0].read_text())
    assert reloaded.allocator_name == "mock"
    assert reloaded.spec == spec
    parsed = json.loads(files[0].read_text())
    assert parsed["schema_version"] == "1.0.0"
    assert parsed["metrics"]["allocate_latency_percentiles"]["max_ns"] >= 0


def test_cli_rejects_unknown_workload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main(
        [
            "--workload",
            "not_a_real_workload",
            "--allocator",
            "mock",
            "--device",
            "cpu",
            "--output",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Unknown workload" in captured.err


def test_cli_rejects_unknown_allocator_with_empty_registry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    REGISTRY.pop("mock", None)
    exit_code = cli_main(
        [
            "--workload",
            "uniform_short",
            "--allocator",
            "does_not_exist",
            "--device",
            "cpu",
            "--output",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Unknown allocator" in captured.err


def test_cli_runs_to_completion_with_registered_allocator(
    registered_mock_allocator: object,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del registered_mock_allocator
    exit_code = cli_main(
        [
            "--workload",
            "uniform_short",
            "--allocator",
            "mock",
            "--device",
            "cpu",
            "--output",
            str(tmp_path),
            "--seed",
            "7",
            "--notes",
            "smoke",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "wrote BenchmarkRun" in captured.out
    assert list((tmp_path / "mock" / "uniform_short").glob("*.json"))


def test_cli_negative_path_via_subprocess(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cachepawl.benchmarks.run",
            "--workload",
            "uniform_short",
            "--allocator",
            "does_not_exist",
            "--device",
            "cpu",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Unknown allocator" in result.stderr
