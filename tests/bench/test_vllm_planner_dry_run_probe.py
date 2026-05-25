"""Tests for vLLM planner dry-run probe artifact generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = "benchmarks/scripts/create_vllm_planner_dry_run_probe.py"


def test_planner_dry_run_probe_writes_artifact_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "translated_runtime_cache_config.json").write_text(
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
        json.dumps({"available_gpu_memory_for_kv_cache": 4096}) + "\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            "2026-05-25T00:00:00+00:00",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    dry_run = json.loads((output_dir / "dry_run_result.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    summary = (output_dir / "summary.md").read_text()
    readme = (output_dir / "README.md").read_text()
    assert dry_run["status"] == "planner_dry_run_available"
    assert dry_run["vanilla_observed_reserved_bytes"] == 1000
    assert dry_run["vanilla_observed_useful_bytes"] == 800
    assert dry_run["cachepawl_proposed_reserved_bytes"] == 800
    assert dry_run["estimated_savings_bytes"] == 200
    assert dry_run["safe_for_advisory_only"] is True
    assert dry_run["returned_to_vllm"] is False
    assert dry_run["vllm_behavior_changed"] is False
    assert manifest["returned_to_vllm"] is False
    assert manifest["vllm_behavior_changed"] is False
    assert "does not return that proposal to vLLM" in summary
    assert "non-mutating Cachepawl proposed planner view" in readme
