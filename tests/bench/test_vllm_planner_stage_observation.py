"""Tests for vLLM planner-stage observation artifacts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = "benchmarks/scripts/capture_vllm_planner_stage_observation.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("capture_vllm_planner_stage_observation", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_deepcopy_failure_classification_is_preserved_in_blocker(tmp_path: Path) -> None:
    module = _load_script_module()
    child_payload = {
        "status": "planner_stage_replay_failed",
        "inputs_reached": True,
        "replay_failed": True,
        "deepcopy_failed": True,
        "get_kv_cache_configs_called": False,
        "unsupported_reason": (
            "deepcopy failed before planner replay: TypeError: "
            "BasevLLMParameter.__new__() takes 2 positional arguments but 3 were given"
        ),
        "object_access": {
            "vllm_config": True,
            "kv_cache_specs": True,
            "available_memory": True,
            "get_kv_cache_configs_called": False,
            "deepcopied_real_inputs": False,
            "returned_to_vllm": False,
            "long_lived_serve": False,
            "allocator_replacement": False,
            "monkeypatching": False,
            "vllm_source_modified": False,
            "scheduler_mutation": False,
            "worker_layout_mutation": False,
        },
        "input_availability": {
            "vllm_config": True,
            "kv_cache_specs": True,
            "available_memory": True,
            "available_memory_exact_for_worker_count": True,
        },
    }
    child_result = {
        "status": "failed",
        "reason": child_payload["unsupported_reason"],
        "payload": child_payload,
    }

    blocker = module._blocker(
        timestamp="2026-05-26T00:00:00+00:00",
        model="fake/model",
        reason=str(child_result["reason"]),
        vllm_version="0.21.0",
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        max_num_seqs=1,
        timeout_seconds=1200,
        trust_remote_code=False,
        extra_metadata=child_result,
    )
    module._write_blocker(tmp_path, blocker)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    metadata = json.loads((tmp_path / "raw_safe_metadata.json").read_text())
    assert manifest["object_access"]["vllm_config"] is True
    assert manifest["object_access"]["get_kv_cache_configs_called"] is False
    assert manifest["input_availability"]["available_memory"] is True
    assert metadata["payload"]["deepcopy_failed"] is True
    assert "BasevLLMParameter.__new__" in manifest["reason"]


def test_inputs_reached_before_direct_replay_failure_are_recorded(tmp_path: Path) -> None:
    module = _load_script_module()
    child_payload = {
        "status": "planner_stage_replay_failed",
        "inputs_reached": True,
        "replay_failed": True,
        "deepcopy_failed": False,
        "get_kv_cache_configs_called": True,
        "unsupported_reason": "direct planner replay failed after real planner inputs were reached",
        "object_access": {
            "vllm_config": True,
            "kv_cache_specs": True,
            "available_memory": True,
            "get_kv_cache_configs_called": True,
            "deepcopied_real_inputs": False,
            "returned_to_vllm": False,
            "long_lived_serve": False,
            "allocator_replacement": False,
            "monkeypatching": False,
            "vllm_source_modified": False,
            "scheduler_mutation": False,
            "worker_layout_mutation": False,
        },
        "input_availability": {
            "vllm_config": True,
            "kv_cache_specs": True,
            "available_memory": True,
            "available_memory_exact_for_worker_count": True,
        },
        "runtime_num_blocks": 284256,
    }
    blocker = module._blocker(
        timestamp="2026-05-26T00:00:00+00:00",
        model="fake/model",
        reason=str(child_payload["unsupported_reason"]),
        vllm_version="0.21.0",
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        max_num_seqs=1,
        timeout_seconds=1200,
        trust_remote_code=False,
        extra_metadata={"status": "failed", "payload": child_payload},
    )
    module._write_blocker(tmp_path, blocker)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    readme = (tmp_path / "README.md").read_text()
    assert manifest["object_access"]["vllm_config"] is True
    assert manifest["object_access"]["kv_cache_specs"] is True
    assert manifest["object_access"]["available_memory"] is True
    assert manifest["object_access"]["get_kv_cache_configs_called"] is True
    assert manifest["object_access"]["returned_to_vllm"] is False
    assert manifest["input_availability"]["available_memory_exact_for_worker_count"] is True
    assert "`inputs_reached`: True" in readme
    assert "`get_kv_cache_configs_called`: True" in readme


def test_successful_direct_replay_writer_records_translated_artifact(tmp_path: Path) -> None:
    module = _load_script_module()
    (tmp_path / "blocker.json").write_text("{}\n")
    child_payload = {
        "status": "planner_stage_translation",
        "function_path": "vllm.v1.core.kv_cache_utils.get_kv_cache_configs",
        "input_paths": {
            "vllm_config": "core.vllm_config",
            "kv_cache_specs": "core.model_executor.get_kv_cache_specs()",
            "available_memory": "core.available_gpu_memory_for_kv_cache",
        },
        "observation_mode": "post_init_direct_replay_on_real_inputs",
        "get_kv_cache_configs_signature": "(vllm_config, kv_cache_specs, available_memory)",
        "inputs_reached": True,
        "get_kv_cache_configs_called": True,
        "deepcopy_failed": False,
        "vllm_config_type": "VllmConfig",
        "kv_cache_specs_worker_count": 1,
        "kv_cache_specs_layer_counts": [28],
        "kv_cache_spec_type_counts": {"FullAttentionSpec": 28},
        "sample_layer_names": ["layers.0.self_attn"],
        "available_memory": [2910781440],
        "planner_output_config_count": 1,
        "planner_output_num_blocks": [284256],
        "runtime_num_blocks": 284256,
        "runtime_num_blocks_after_replay": 284256,
        "runtime_changed_during_replay": False,
        "planner_matches_runtime_scheduler": True,
        "translated_planner_stage_config": {
            "num_blocks": 284256,
            "cache_groups": [],
            "kv_cache_tensors": [],
        },
    }
    payload = module._success_payload(
        timestamp="2026-05-26T00:00:00+00:00",
        model="fake/model",
        vllm_version="0.21.0",
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        max_num_seqs=1,
        timeout_seconds=1200,
        trust_remote_code=False,
        child_payload=child_payload,
        child_metadata={"command": "python -c ..."},
    )
    module._write_success(tmp_path, payload)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    translated = json.loads((tmp_path / "translated_planner_stage_config.json").read_text())
    metadata = json.loads((tmp_path / "raw_safe_metadata.json").read_text())
    assert manifest["status"] == "planner_stage_translation"
    assert manifest["object_access"]["deepcopied_real_inputs"] is False
    assert manifest["object_access"]["direct_real_inputs"] is True
    assert manifest["object_access"]["returned_to_vllm"] is False
    assert metadata["observation_mode"] == "post_init_direct_replay_on_real_inputs"
    assert metadata["runtime_changed_during_replay"] is False
    assert translated["num_blocks"] == 284256
    assert not (tmp_path / "blocker.json").exists()
