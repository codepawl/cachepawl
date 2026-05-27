"""Tests for vLLM Mamba/attention contract artifact writing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path("benchmarks/scripts/capture_vllm_mamba_attention_contract_observation.py")


def test_mamba_attention_contract_writer_records_complete_observation(tmp_path: Path) -> None:
    script = _load_script_module()
    payload = script.create_mamba_attention_contract_artifact_payload(
        timestamp="2026-05-27T00:00:00+00:00",
        model="Zyphra/Zamba2-2.7B-instruct",
        prompt="Count from one to four:",
        vllm_version="0.21.0",
        max_new_tokens=8,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        max_num_seqs=1,
        timeout_seconds=1200,
        trust_remote_code=False,
        child_payload=_child_payload(field_blockers=()),
        child_metadata={"command": "python -c observe", "status": "completed"},
    )

    script.write_mamba_attention_contract_artifact(tmp_path, payload)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    report = json.loads((tmp_path / "mamba_attention_contract_report.json").read_text())
    blockers = json.loads((tmp_path / "field_level_blockers.json").read_text())
    summary = (tmp_path / "summary.md").read_text()
    readme = (tmp_path / "README.md").read_text()

    assert manifest["artifact"] == "vllm-mamba-attention-contract-observation"
    assert manifest["classification"] == "mamba_attention_contract_observation_complete"
    assert manifest["field_blocker_count"] == 0
    assert report["request_id"] == "0"
    assert report["tensor_serialization"] is False
    assert blockers["field_level_blockers"] == []
    assert "Request id: `0`" in summary
    assert "No tensors or large model objects were serialized" in summary
    assert "mamba_attention_contract_report.json" in readme


def test_mamba_attention_contract_writer_records_field_blockers(tmp_path: Path) -> None:
    script = _load_script_module()
    blocker: dict[str, object] = {
        "name": "mamba_state_tensor_contract",
        "status": "blocked",
        "evidence": {"max_mamba_related_tensor_count": 0},
        "blocker_reason": "Mamba state tensors were not safely reachable",
    }
    payload = script.create_mamba_attention_contract_artifact_payload(
        timestamp="2026-05-27T00:00:00+00:00",
        model="Zyphra/Zamba2-2.7B-instruct",
        prompt="Count from one to four:",
        vllm_version="0.21.0",
        max_new_tokens=8,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        max_num_seqs=1,
        timeout_seconds=1200,
        trust_remote_code=False,
        child_payload=_child_payload(field_blockers=(blocker,)),
        child_metadata={"command": "python -c observe", "status": "completed"},
    )

    script.write_mamba_attention_contract_artifact(tmp_path, payload)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    blockers = json.loads((tmp_path / "field_level_blockers.json").read_text())

    assert manifest["classification"] == (
        "mamba_attention_contract_observation_with_field_blockers"
    )
    assert [item["name"] for item in blockers["field_level_blockers"]] == [
        "mamba_state_tensor_contract"
    ]


def test_mamba_attention_contract_rejects_unbounded_workload() -> None:
    script = _load_script_module()

    try:
        script.capture_mamba_attention_contract_observation(
            output_dir=Path("/tmp/cachepawl-mamba-attention-test"),
            timestamp="2026-05-27T00:00:00+00:00",
            model="Zyphra/Zamba2-2.7B-instruct",
            prompt="Count from one to four:",
            max_new_tokens=9,
            max_model_len=4096,
            gpu_memory_utilization=0.7,
            max_num_seqs=1,
            timeout_seconds=1200,
        )
    except ValueError as exc:
        assert str(exc) == "max_new_tokens must be between 1 and 8"
    else:
        raise AssertionError("expected ValueError")


def _child_payload(*, field_blockers: tuple[dict[str, object], ...]) -> dict[str, object]:
    fields = [
        {
            "name": "mamba_state_index_contract",
            "status": "observed",
            "evidence": {"phases_with_request_state_index": ("after_first_step",)},
            "blocker_reason": None,
        },
        {
            "name": "attention_block_table_view_contract",
            "status": "observed",
            "evidence": {"max_block_table_tensor_count": 2},
            "blocker_reason": None,
        },
        {
            "name": "attention_metadata_builder_contract",
            "status": "observed",
            "evidence": {"max_attention_group_count": 7},
            "blocker_reason": None,
        },
        {
            "name": "mamba_state_tensor_contract",
            "status": "observed" if not field_blockers else "blocked",
            "evidence": {"max_mamba_related_tensor_count": 1},
            "blocker_reason": None if not field_blockers else "no tensors",
        },
    ]
    return {
        "status": "mamba_attention_contract_observation",
        "runtime_path": "LLM.llm_engine.engine_core.engine_core",
        "request_id": "0",
        "prompt": "Count from one to four:",
        "max_new_tokens": 8,
        "snapshots": [
            {
                "phase": "after_first_step",
                "totals": {
                    "mamba_state_idx_contains_request": True,
                    "block_table_tensor_count": 2,
                    "attention_group_count": 7,
                    "mamba_related_tensor_count": 1,
                },
            }
        ],
        "fields": fields,
        "field_level_blockers": list(field_blockers),
        "raw_safe_metadata": {"scheduler_type": "FakeScheduler"},
        "object_access": {
            "runtime_contract_objects_reached": True,
            "live_request_scheduled": True,
            "long_lived_serve": False,
            "allocator_replacement": False,
            "monkeypatching": False,
            "vllm_source_modified": False,
            "scheduler_mutation": False,
            "worker_layout_mutation": False,
            "returned_to_vllm": False,
            "controlled_substitution": False,
            "tensor_serialization": False,
        },
    }


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "capture_vllm_mamba_attention_contract_observation_for_test",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
