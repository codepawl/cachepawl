"""Tests for vLLM runtime cache diagnostic artifact generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = "benchmarks/scripts/create_vllm_cache_diagnostic.py"


def test_cache_diagnostic_script_writes_report_summary_and_manifest(tmp_path: Path) -> None:
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
    report = json.loads((output_dir / "report.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    summary = (output_dir / "summary.md").read_text()
    assert report["primary_classification"] == "planner_advisory_available"
    assert report["observed_reserved_bytes"] == 1000
    assert report["observed_useful_bytes"] == 800
    assert report["advisory_savings_bytes"] == 200
    assert manifest["vllm_behavior_changed"] is False
    assert manifest["observe_only"] is True
    assert "advisory and diagnostic only" in summary
    assert "runtime improvement still requires" in summary
