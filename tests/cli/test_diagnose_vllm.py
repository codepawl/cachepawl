"""Tests for the artifact-input vLLM diagnostic CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cachepawl


def test_cachepawl_version_flag_reports_package_version() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "cachepawl.cli", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == f"cachepawl {cachepawl.__version__}\n"
    assert completed.stderr == ""


def test_diagnose_vllm_writes_report_summary_and_manifest(tmp_path: Path) -> None:
    translated_path, raw_metadata_path = _write_fixture_inputs(tmp_path / "input")
    output_dir = tmp_path / "output"

    completed = _run_cli(
        "--translated-cache-config",
        str(translated_path),
        "--raw-safe-metadata",
        str(raw_metadata_path),
        "--output-dir",
        str(output_dir),
        "--timestamp",
        "2026-05-26T00:00:00+00:00",
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((output_dir / "report.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    summary = (output_dir / "summary.md").read_text()
    assert report["classification"] == "planner_advisory_available"
    assert report["observed_reserved_bytes"] == 1000
    assert report["observed_useful_bytes"] == 800
    assert report["cachepawl_recommended_bytes"] == 800
    assert report["advisory_savings_bytes"] == 200
    assert report["overestimation_ratio"] == 1.25
    assert report["wasted_fraction"] == 0.2
    assert report["cache_group_count"] == 1
    assert report["cache_tensor_count"] == 1
    assert report["layer_count"] == 1
    assert report["num_blocks"] == 10
    assert report["advisory_only"] is True
    assert report["vllm_behavior_changed"] is False
    assert report["runtime_mutation"] is False
    assert report["allocator_replacement"] is False
    assert report["runtime_savings_require_future_mutation_hook"] is True
    assert report["dry_run"]["status"] == "planner_dry_run_available"
    assert manifest["vllm_required"] is False
    assert manifest["cuda_required"] is False
    assert manifest["gpu_required"] is False
    assert manifest["nvml_required"] is False
    assert manifest["model_loaded"] is False
    assert "advisory-only" in summary
    assert "does not rerun vLLM" in summary
    assert "replace allocators" in summary
    assert "Runtime savings require a future mutation hook" in summary


def test_diagnose_vllm_missing_translated_config_reports_clear_error(tmp_path: Path) -> None:
    completed = _run_cli(
        "--translated-cache-config",
        str(tmp_path / "missing.json"),
        "--output-dir",
        str(tmp_path / "output"),
    )

    assert completed.returncode == 2
    assert "missing translated config file" in completed.stderr


def test_diagnose_vllm_invalid_json_reports_clear_error(tmp_path: Path) -> None:
    translated_path = tmp_path / "translated_runtime_cache_config.json"
    translated_path.write_text("{not json\n")

    completed = _run_cli(
        "--translated-cache-config",
        str(translated_path),
        "--output-dir",
        str(tmp_path / "output"),
    )

    assert completed.returncode == 2
    assert "invalid JSON" in completed.stderr
    assert str(translated_path) in completed.stderr


def test_diagnose_vllm_unsupported_schema_reports_clear_error(tmp_path: Path) -> None:
    translated_path = tmp_path / "translated_runtime_cache_config.json"
    translated_path.write_text(json.dumps({"num_blocks": "ten"}) + "\n")

    completed = _run_cli(
        "--translated-cache-config",
        str(translated_path),
        "--output-dir",
        str(tmp_path / "output"),
    )

    assert completed.returncode == 2
    assert "unsupported translated config schema" in completed.stderr


def test_diagnose_vllm_output_is_deterministic(tmp_path: Path) -> None:
    translated_path, raw_metadata_path = _write_fixture_inputs(tmp_path / "input")
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    args = [
        "--translated-cache-config",
        str(translated_path),
        "--raw-safe-metadata",
        str(raw_metadata_path),
        "--timestamp",
        "2026-05-26T00:00:00+00:00",
    ]

    completed_a = _run_cli(*args, "--output-dir", str(output_a))
    completed_b = _run_cli(*args, "--output-dir", str(output_b))

    assert completed_a.returncode == 0, completed_a.stderr
    assert completed_b.returncode == 0, completed_b.stderr
    assert (output_a / "report.json").read_text() == (output_b / "report.json").read_text()
    assert (output_a / "summary.md").read_text() == (output_b / "summary.md").read_text()

    manifest_a = json.loads((output_a / "manifest.json").read_text())
    manifest_b = json.loads((output_b / "manifest.json").read_text())
    manifest_a["outputs"] = manifest_b["outputs"]
    assert manifest_a == manifest_b


def test_diagnose_vllm_summary_only_prints_markdown_and_writes_files(tmp_path: Path) -> None:
    translated_path, raw_metadata_path = _write_fixture_inputs(tmp_path / "input")
    output_dir = tmp_path / "output"

    completed = _run_cli(
        "--translated-cache-config",
        str(translated_path),
        "--raw-safe-metadata",
        str(raw_metadata_path),
        "--output-dir",
        str(output_dir),
        "--summary-only",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.startswith("# vLLM Runtime Cache Diagnostic")
    assert "`wasted_fraction`: 0.2" in completed.stdout
    assert (output_dir / "report.json").is_file()
    assert (output_dir / "summary.md").is_file()
    assert (output_dir / "manifest.json").is_file()


def test_diagnose_vllm_summary_only_json_is_deterministic(tmp_path: Path) -> None:
    translated_path, raw_metadata_path = _write_fixture_inputs(tmp_path / "input")
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    args = [
        "--translated-cache-config",
        str(translated_path),
        "--raw-safe-metadata",
        str(raw_metadata_path),
        "--summary-only",
        "--format",
        "json",
        "--timestamp",
        "2026-05-26T00:00:00+00:00",
    ]

    completed_a = _run_cli(*args, "--output-dir", str(output_a))
    completed_b = _run_cli(*args, "--output-dir", str(output_b))

    assert completed_a.returncode == 0, completed_a.stderr
    assert completed_b.returncode == 0, completed_b.stderr
    assert completed_a.stdout == completed_b.stdout
    report = json.loads(completed_a.stdout)
    assert report["classification"] == "planner_advisory_available"
    assert report["wasted_fraction"] == 0.2


def test_diagnose_vllm_thresholds_pass_when_metric_equals_threshold(
    tmp_path: Path,
) -> None:
    translated_path, raw_metadata_path = _write_fixture_inputs(tmp_path / "input")

    completed = _run_cli(
        "--translated-cache-config",
        str(translated_path),
        "--raw-safe-metadata",
        str(raw_metadata_path),
        "--output-dir",
        str(tmp_path / "output"),
        "--fail-on-waste-fraction",
        "0.2",
        "--fail-on-overestimation-ratio",
        "1.25",
    )

    assert completed.returncode == 0, completed.stderr


def test_diagnose_vllm_waste_threshold_failure_writes_outputs(
    tmp_path: Path,
) -> None:
    translated_path, raw_metadata_path = _write_fixture_inputs(tmp_path / "input")
    output_dir = tmp_path / "output"

    completed = _run_cli(
        "--translated-cache-config",
        str(translated_path),
        "--raw-safe-metadata",
        str(raw_metadata_path),
        "--output-dir",
        str(output_dir),
        "--fail-on-waste-fraction",
        "0.19",
    )

    assert completed.returncode == 1
    assert "threshold failed: wasted_fraction 0.2 > 0.19" in completed.stderr
    assert (output_dir / "report.json").is_file()
    assert (output_dir / "summary.md").is_file()
    assert (output_dir / "manifest.json").is_file()


def test_diagnose_vllm_overestimation_threshold_failure_writes_outputs(
    tmp_path: Path,
) -> None:
    translated_path, raw_metadata_path = _write_fixture_inputs(tmp_path / "input")
    output_dir = tmp_path / "output"

    completed = _run_cli(
        "--translated-cache-config",
        str(translated_path),
        "--raw-safe-metadata",
        str(raw_metadata_path),
        "--output-dir",
        str(output_dir),
        "--fail-on-overestimation-ratio",
        "1.24",
    )

    assert completed.returncode == 1
    assert "threshold failed: overestimation_ratio 1.25 > 1.24" in completed.stderr
    assert (output_dir / "report.json").is_file()


def test_diagnose_vllm_negative_threshold_reports_arg_error(tmp_path: Path) -> None:
    translated_path, _ = _write_fixture_inputs(tmp_path / "input")

    completed = _run_cli(
        "--translated-cache-config",
        str(translated_path),
        "--output-dir",
        str(tmp_path / "output"),
        "--fail-on-waste-fraction",
        "-0.1",
    )

    assert completed.returncode == 2
    assert "must be non-negative" in completed.stderr


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cachepawl.cli", "diagnose-vllm", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _write_fixture_inputs(input_dir: Path) -> tuple[Path, Path]:
    input_dir.mkdir()
    translated_path = input_dir / "translated_runtime_cache_config.json"
    raw_metadata_path = input_dir / "raw_safe_metadata.json"
    translated_path.write_text(
        json.dumps(
            {
                "num_blocks": 10,
                "group_count": 1,
                "layer_count": 1,
                "groups": [
                    {
                        "group_index": 0,
                        "layer_count": 1,
                        "layer_names": ["attn.0"],
                        "cache_spec": {
                            "cache_kind": "attention",
                            "block_size": 16,
                            "page_size_bytes": 100,
                            "useful_bytes": 80,
                        },
                    }
                ],
                "tensors": [{"size_bytes": 1000, "shared_by": ["attn.0"]}],
                "total_useful_bytes": 80,
            }
        )
        + "\n"
    )
    raw_metadata_path.write_text(json.dumps({"available_gpu_memory_for_kv_cache": 4096}) + "\n")
    return translated_path, raw_metadata_path
