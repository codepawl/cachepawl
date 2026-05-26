"""Tests for vLLM planner-stage advisory diff artifact generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = "benchmarks/scripts/create_vllm_planner_stage_advisory_diff.py"


def test_planner_stage_advisory_diff_writes_artifact_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    translated_path = input_dir / "translated_planner_stage_config.json"
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
    (input_dir / "raw_safe_metadata.json").write_text(
        json.dumps(
            {
                "planner_matches_runtime_scheduler": True,
                "runtime_changed_during_replay": False,
            }
        )
        + "\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--translated-planner-stage-config",
            str(translated_path),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            "2026-05-26T00:00:00+00:00",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    diff = json.loads((output_dir / "diff_report.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    group_diff = json.loads((output_dir / "group_level_diff.json").read_text())
    summary = (output_dir / "summary.md").read_text()
    readme = (output_dir / "README.md").read_text()
    assert diff["status"] == "planner_stage_advisory_diff_available"
    assert diff["vanilla_reserved_bytes"] == 1000
    assert diff["vanilla_useful_bytes"] == 800
    assert diff["cachepawl_proposed_reserved_bytes"] == 800
    assert diff["estimated_savings_bytes"] == 200
    assert diff["non_mutating"] is True
    assert diff["returned_to_vllm"] is False
    assert diff["vllm_behavior_changed"] is False
    assert diff["parity_status"]["planner_matches_runtime_scheduler"] is True
    assert manifest["returned_to_vllm"] is False
    assert manifest["vllm_required"] is False
    assert group_diff[0]["estimated_savings_bytes"] == 200
    assert "does not return that proposal to vLLM" in summary
    assert "does not change vanilla vLLM" in readme
