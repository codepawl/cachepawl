"""Resume-from-disk tests for the sweep runner.

`run_sweep` should pick up where a killed sweep left off by reading the
per-cell JSONs that the prior run wrote to ``{output_dir}/runs/...``.
Corrupt or schema-mismatched files are treated as missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cachepawl.benchmarks.compare.sweep import SweepConfig, run_sweep


def _smoke_config(output_dir: Path) -> SweepConfig:
    """Minimal 2-cell sweep on cpu: 1 variant x 1 workload x 1 spec x 1 size x 2 seeds.

    Small enough that re-running is cheap; large enough to verify the
    resume skip path skips one cell and runs the other.
    """

    from cachepawl.benchmarks.compare.sweep import (
        DEFAULT_VARIANTS,
        QUICK_MODEL_SPEC_NAMES,
        QUICK_TOTAL_BYTES_OPTIONS,
        QUICK_WORKLOAD_NAMES,
        SMOKE_NUM_REQUESTS,
    )

    return SweepConfig(
        variants=(DEFAULT_VARIANTS[0],),
        workload_names=QUICK_WORKLOAD_NAMES,
        model_spec_names=QUICK_MODEL_SPEC_NAMES,
        total_bytes_options=QUICK_TOTAL_BYTES_OPTIONS,
        device="cpu",
        output_dir=output_dir,
        seed_replicates=2,
        smoke_num_requests=SMOKE_NUM_REQUESTS,
    )


def test_resume_skips_cells_with_existing_json(tmp_path: Path) -> None:
    """First run populates runs/; second run reads them back without re-executing."""

    config = _smoke_config(tmp_path)

    # First run computes both cells.
    first = run_sweep(config)
    assert len(first.runs) == 2
    assert first.metadata.n_cells_succeeded == 2

    # The per-cell JSONs exist on disk.
    runs_dir = tmp_path / "runs"
    json_files = sorted(runs_dir.rglob("*.json"))
    assert len(json_files) == 2

    # Second run: resume. Should reuse both JSONs without re-executing.
    second = run_sweep(config)
    assert len(second.runs) == 2
    assert second.metadata.n_cells_succeeded == 2
    # Same aggregated counters as the first run.
    assert [r.metrics.oom_count for r in second.runs] == [r.metrics.oom_count for r in first.runs]


def test_resume_reruns_corrupt_cell(tmp_path: Path) -> None:
    """A corrupt JSON file is treated as missing and the cell re-runs."""

    config = _smoke_config(tmp_path)

    # First run populates runs/.
    first = run_sweep(config)
    assert len(first.runs) == 2

    # Corrupt one of the per-cell JSONs.
    json_files = sorted((tmp_path / "runs").rglob("*.json"))
    corrupt = json_files[0]
    corrupt.write_text("not valid json {{")

    # Second run: should skip the intact file, re-run the corrupt one.
    second = run_sweep(config)
    assert len(second.runs) == 2
    # The corrupt file got rewritten with valid JSON.
    assert corrupt.read_text().startswith("{")


def test_resume_with_partial_runs_dir(tmp_path: Path) -> None:
    """Only one of two cell JSONs present: resume runs the missing one."""

    config = _smoke_config(tmp_path)

    # First run populates both, then delete one.
    run_sweep(config)
    json_files = sorted((tmp_path / "runs").rglob("*.json"))
    json_files[0].unlink()
    assert len(list((tmp_path / "runs").rglob("*.json"))) == 1

    # Resume: should write the deleted one back.
    second = run_sweep(config)
    assert len(second.runs) == 2
    assert len(list((tmp_path / "runs").rglob("*.json"))) == 2


def test_resume_emits_resumed_label_in_progress_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Resumed cells print 'RESUMED' so a tail-of-log shows which cells reused state."""

    config = _smoke_config(tmp_path)
    # First run populates runs/.
    run_sweep(config)
    capsys.readouterr()  # discard the first run's output

    second = run_sweep(config)
    assert second.metadata.n_cells_succeeded == 2
    out = capsys.readouterr().out
    assert "RESUMED" in out, f"expected RESUMED in second run output, got:\n{out}"
