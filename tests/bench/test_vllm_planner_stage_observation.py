"""Tests for vLLM planner-stage observation artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = "benchmarks/scripts/capture_vllm_planner_stage_observation.py"


def test_planner_stage_observation_records_blocker_without_vllm(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--output-dir",
            str(tmp_path),
            "--timestamp",
            "2026-05-25T00:00:00+00:00",
            "--timeout-seconds",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    blocker = json.loads((tmp_path / "blocker.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    readme = (tmp_path / "README.md").read_text()
    assert blocker["manifest"]["reason"] == "vllm is not installed in the active Python environment"
    assert manifest["status"] == "blocked"
    assert manifest["object_access"]["get_kv_cache_configs_called"] is False
    assert manifest["object_access"]["returned_to_vllm"] is False
    assert "get_kv_cache_configs" in readme
    assert not (tmp_path / "translated_planner_stage_config.json").exists()
