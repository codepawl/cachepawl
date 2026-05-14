"""CLI smoke test: --quick --smoke --device cpu exits 0 in under a minute."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_smoke_run_creates_expected_outputs(tmp_path: Path) -> None:
    """Subprocess invocation produces every artifact the runner promises."""

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cachepawl.benchmarks.compare",
            "--quick",
            "--smoke",
            "--device",
            "cpu",
            "--output",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "sweep complete:" in result.stdout
    assert "0 failures" in result.stdout
    assert (tmp_path / "SWEEP_METADATA.json").exists()
    assert (tmp_path / "aggregated.json").exists()
    assert (tmp_path / "aggregated_deterministic.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "figures" / "fragmentation_vs_workload.png").exists()
    assert (tmp_path / "figures" / "padding_waste_vs_state_size.png").exists()
    runs_root = tmp_path / "runs"
    assert runs_root.is_dir()
    cell_files = list(runs_root.rglob("*.json"))
    assert len(cell_files) == 1, f"expected exactly one per-cell JSON, got {len(cell_files)}"


def test_cli_smoke_run_deterministic_subset_byte_identical(tmp_path: Path) -> None:
    """Two reruns at the same seed produce byte-identical deterministic JSON."""

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cachepawl.benchmarks.compare",
                "--quick",
                "--smoke",
                "--device",
                "cpu",
                "--output",
                str(out),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
    text_a = (out_a / "aggregated_deterministic.json").read_text()
    text_b = (out_b / "aggregated_deterministic.json").read_text()
    assert text_a == text_b
