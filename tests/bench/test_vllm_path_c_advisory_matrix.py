"""Tests for vLLM Path C advisory matrix table generation."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path("benchmarks/scripts/run_vllm_path_c_advisory_matrix.py")


def test_matrix_config_parsing_and_table_generation(tmp_path: Path) -> None:
    script = _load_script_module()
    config_path = tmp_path / "config_matrix.json"
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "evaluation"
    config_path.write_text(
        json.dumps(
            {
                "artifact": "vllm-path-c-advisory-config-matrix",
                "model": "Zyphra/Zamba2-2.7B-instruct",
                "vllm_version": "0.21.0",
                "pinned_venv_path": "/tmp/vllm-env",
                "results_dir": str(results_dir),
                "timeout_seconds": 1200,
                "trust_remote_code": False,
                "matrix": {
                    "max_model_len": [2048, 4096],
                    "gpu_memory_utilization": [0.6, 0.7],
                    "max_num_seqs": [1],
                },
            }
        )
        + "\n"
    )
    config = script.load_matrix_config(config_path)
    completed_point = script.MatrixPoint(
        model="Zyphra/Zamba2-2.7B-instruct",
        max_model_len=2048,
        gpu_memory_utilization=0.6,
        max_num_seqs=1,
    )
    diff_report_path = (
        results_dir
        / "Zyphra_Zamba2-2.7B-instruct__len2048__gpu0p6__seqs1"
        / "planner-stage-advisory-diff"
        / "diff_report.json"
    )
    diff_report_path.parent.mkdir(parents=True)
    diff_report_path.write_text(json.dumps(_fake_diff_report()) + "\n")

    rows = script.matrix_rows_from_config(
        config=config,
        results_dir=results_dir,
        existing_baseline_diff_report=None,
    )
    script.write_matrix_artifacts(
        rows=rows,
        config=config,
        output_dir=output_dir,
        results_dir=results_dir,
    )

    assert len(rows) == 4
    completed = next(row for row in rows if row.point == completed_point)
    assert completed.status == "completed"
    assert completed.vanilla_reserved_bytes == 1000
    assert completed.estimated_savings_bytes == 200
    pending = [row for row in rows if row.status == "pending_not_run"]
    assert len(pending) == 3
    assert {row.blocker for row in pending} == {"matrix point not run"}

    with (output_dir / "matrix_table.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 4
    assert csv_rows[0]["model"] == "Zyphra/Zamba2-2.7B-instruct"
    assert csv_rows[0]["status"] == "completed"
    assert "matrix point not run" in (output_dir / "matrix_table.md").read_text()
    assert (
        results_dir / "Zyphra_Zamba2-2.7B-instruct__len2048__gpu0p7__seqs1" / "manifest.json"
    ).exists()


def test_existing_baseline_populates_matching_matrix_point(tmp_path: Path) -> None:
    script = _load_script_module()
    config_path = tmp_path / "config_matrix.json"
    results_dir = tmp_path / "results"
    baseline_path = tmp_path / "baseline_diff_report.json"
    config_path.write_text(
        json.dumps(
            {
                "artifact": "vllm-path-c-advisory-config-matrix",
                "model": "Zyphra/Zamba2-2.7B-instruct",
                "vllm_version": "0.21.0",
                "pinned_venv_path": "/tmp/vllm-env",
                "results_dir": str(results_dir),
                "matrix": {
                    "max_model_len": [4096],
                    "gpu_memory_utilization": [0.7],
                    "max_num_seqs": [1],
                },
            }
        )
        + "\n"
    )
    baseline_path.write_text(json.dumps(_fake_diff_report()) + "\n")

    rows = script.matrix_rows_from_config(
        config=script.load_matrix_config(config_path),
        results_dir=results_dir,
        existing_baseline_diff_report=baseline_path,
    )

    assert len(rows) == 1
    assert rows[0].status == "completed_existing_baseline"
    assert rows[0].cache_group_count == 7
    assert rows[0].blocker == ""


def _fake_diff_report() -> dict[str, object]:
    return {
        "status": "planner_stage_advisory_diff_available",
        "vanilla_reserved_bytes": 1000,
        "vanilla_useful_bytes": 800,
        "cachepawl_proposed_reserved_bytes": 800,
        "estimated_savings_bytes": 200,
        "overestimation_ratio": 1.25,
        "wasted_fraction": 0.2,
        "num_blocks": 10,
        "cache_group_count": 7,
        "cache_tensor_count": 9,
        "layer_count": 63,
    }


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_vllm_path_c_advisory_matrix_for_test",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
