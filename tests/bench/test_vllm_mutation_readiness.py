"""Tests for vLLM mutation-readiness artifact generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = "benchmarks/scripts/create_vllm_mutation_readiness.py"


def test_mutation_readiness_writes_artifact_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    planner_path = input_dir / "translated_planner_stage_config.json"
    diff_path = input_dir / "diff_report.json"
    group_diff_path = input_dir / "group_level_diff.json"
    planner_path.write_text(json.dumps(_planner_fixture()) + "\n")
    diff_path.write_text(json.dumps(_diff_fixture()) + "\n")
    group_diff_path.write_text(json.dumps(_group_diff_fixture()) + "\n")

    completed = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--translated-planner-stage-config",
            str(planner_path),
            "--diff-report",
            str(diff_path),
            "--group-level-diff",
            str(group_diff_path),
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
    report = json.loads((output_dir / "readiness_report.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    summary = (output_dir / "summary.md").read_text()
    readme = (output_dir / "README.md").read_text()
    assert report["classification"] == "advisory_only_recommended"
    assert report["ready_for_controlled_substitution"] is False
    assert report["failed_invariants"] == []
    assert report["blocked_invariants"] == ["mutation_required_missing_fields"]
    assert report["non_mutating"] is True
    assert report["returned_to_vllm"] is False
    assert report["vllm_behavior_changed"] is False
    assert manifest["status"] == "advisory_only_recommended"
    assert manifest["vllm_required"] is False
    assert manifest["gpu_required"] is False
    assert "pre-mutation safety gate" in summary
    assert "does not require vLLM, GPU, or NVML" in readme


def _planner_fixture() -> dict[str, object]:
    return {
        "num_blocks": 10,
        "group_count": 1,
        "layer_count": 1,
        "attention_group_count": 1,
        "mamba_group_count": 0,
        "groups": [
            {
                "group_index": 0,
                "layer_count": 1,
                "layer_names": ["attn.0"],
                "cache_spec": {
                    "cache_kind": "attention",
                    "spec_type": "FullAttentionSpec",
                    "dtype": "torch.bfloat16",
                    "block_size": 16,
                    "page_size_bytes": 100,
                    "useful_bytes": 80,
                },
            }
        ],
        "tensors": [{"size_bytes": 1000, "shared_by": ["attn.0"]}],
        "total_useful_bytes": 80,
    }


def _diff_fixture() -> dict[str, object]:
    return {
        "num_blocks": 10,
        "cache_group_count": 1,
        "cache_tensor_count": 1,
        "layer_count": 1,
        "vanilla_reserved_bytes": 1000,
        "vanilla_useful_bytes": 800,
        "cachepawl_proposed_reserved_bytes": 800,
        "estimated_savings_bytes": 200,
        "missing_fields_that_prevent_mutation": ["controlled substitution hook"],
    }


def _group_diff_fixture() -> list[dict[str, object]]:
    return [
        {
            "group_index": 0,
            "block_size": 16,
            "page_size_bytes": 100,
            "useful_bytes": 80,
            "vanilla_reserved_bytes": 1000,
            "cachepawl_proposed_reserved_bytes": 800,
            "estimated_savings_bytes": 200,
            "overestimation_ratio": 1.25,
            "wasted_fraction": 0.2,
        }
    ]
