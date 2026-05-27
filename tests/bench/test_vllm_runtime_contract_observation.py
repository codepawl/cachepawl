"""Tests for vLLM runtime-contract observation artifact writing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path("benchmarks/scripts/capture_vllm_runtime_contract_observation.py")


def test_runtime_contract_writer_records_field_blockers(tmp_path: Path) -> None:
    script = _load_script_module()
    payload = script.create_runtime_contract_artifact_payload(
        timestamp="2026-05-27T00:00:00+00:00",
        model="Zyphra/Zamba2-2.7B-instruct",
        vllm_version="0.21.0",
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        max_num_seqs=1,
        timeout_seconds=1200,
        trust_remote_code=False,
        child_payload=_child_payload(),
        child_metadata={"command": "python -c observe", "status": "completed"},
    )

    script.write_runtime_contract_artifact(tmp_path, payload)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    report = json.loads((tmp_path / "runtime_contract_report.json").read_text())
    blockers = json.loads((tmp_path / "field_level_blockers.json").read_text())
    summary = (tmp_path / "summary.md").read_text()
    readme = (tmp_path / "README.md").read_text()

    assert manifest["artifact"] == "vllm-runtime-contract-observation"
    assert manifest["classification"] == "runtime_contract_observation_with_field_blockers"
    assert manifest["field_blocker_count"] == 2
    assert report["non_mutating"] is True
    assert report["returned_to_vllm"] is False
    assert report["vllm_behavior_changed"] is False
    assert [item["name"] for item in blockers["field_level_blockers"]] == [
        "request_to_block_assignment",
        "mamba_state_index_attention_view_contract",
    ]
    assert "Field-Level Blockers" in summary
    assert "No vLLM source edits" in summary
    assert "runtime_contract_report.json" in readme


def _child_payload() -> dict[str, object]:
    fields = [
        {
            "name": "scheduler_kv_cache_manager_structure",
            "status": "observed",
            "evidence": {"kv_cache_manager_type": "FakeKVCacheManager"},
            "blocker_reason": None,
        },
        {
            "name": "block_usage_metadata",
            "status": "observed",
            "evidence": {"block_pool_free_blocks": 321},
            "blocker_reason": None,
        },
        {
            "name": "worker_cache_tensor_layout",
            "status": "observed",
            "evidence": {"tensor_summary_count": 1},
            "blocker_reason": None,
        },
        {
            "name": "request_to_block_assignment",
            "status": "blocked",
            "evidence": {"get_block_ids_callable": True},
            "blocker_reason": "get_block_ids exists but needs a live request id",
        },
        {
            "name": "mamba_state_index_attention_view_contract",
            "status": "blocked",
            "evidence": {"state_index_tensor_count": 0},
            "blocker_reason": "Mamba state-index tensors were not safely reachable",
        },
    ]
    return {
        "status": "runtime_contract_observation",
        "runtime_path": "LLM.llm_engine.engine_core.engine_core",
        "scheduler_manager": {"kv_cache_manager_type": "FakeKVCacheManager"},
        "block_usage": {"block_pool_free_blocks": 321},
        "worker_tensors": [{"shape": [2, 3, 4], "stride": [12, 4, 1]}],
        "mamba_attention": {"state_index_tensor_count": 0},
        "fields": fields,
        "field_level_blockers": [fields[3], fields[4]],
        "raw_safe_metadata": {"scheduler_type": "FakeScheduler"},
        "object_access": {
            "runtime_contract_objects_reached": True,
            "long_lived_serve": False,
            "allocator_replacement": False,
            "monkeypatching": False,
            "vllm_source_modified": False,
            "scheduler_mutation": False,
            "worker_layout_mutation": False,
            "returned_to_vllm": False,
            "controlled_substitution": False,
        },
    }


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "capture_vllm_runtime_contract_observation_for_test",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
