"""Tests for vLLM live-request contract observation artifact writing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path("benchmarks/scripts/capture_vllm_live_request_contract_observation.py")


def test_live_request_contract_writer_records_request_and_blocks(tmp_path: Path) -> None:
    script = _load_script_module()
    payload = script.create_live_request_contract_artifact_payload(
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

    script.write_live_request_contract_artifact(tmp_path, payload)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    report = json.loads((tmp_path / "live_request_contract_report.json").read_text())
    blockers = json.loads((tmp_path / "field_level_blockers.json").read_text())
    summary = (tmp_path / "summary.md").read_text()
    readme = (tmp_path / "README.md").read_text()

    assert manifest["artifact"] == "vllm-live-request-contract-observation"
    assert manifest["classification"] == "live_request_contract_observation_complete"
    assert manifest["field_blocker_count"] == 0
    assert report["request_id"] == "0"
    assert report["non_mutating"] is True
    assert report["returned_to_vllm"] is False
    assert report["vllm_behavior_changed"] is False
    assert blockers["field_level_blockers"] == []
    assert "Request id: `0`" in summary
    assert "No vLLM source edits" in summary
    assert "live_request_contract_report.json" in readme


def test_live_request_contract_writer_records_field_blockers(tmp_path: Path) -> None:
    script = _load_script_module()
    request_blocker: dict[str, object] = {
        "name": "request_to_block_assignment",
        "status": "blocked",
        "evidence": {"request_id": "0", "active_total_block_id_counts": ()},
        "blocker_reason": "no non-empty block id assignment was reachable",
    }
    payload = script.create_live_request_contract_artifact_payload(
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
        child_payload=_child_payload(field_blockers=(request_blocker,)),
        child_metadata={"command": "python -c observe", "status": "completed"},
    )

    script.write_live_request_contract_artifact(tmp_path, payload)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    blockers = json.loads((tmp_path / "field_level_blockers.json").read_text())

    assert manifest["classification"] == "live_request_contract_observation_with_field_blockers"
    assert manifest["field_blocker_count"] == 1
    assert [item["name"] for item in blockers["field_level_blockers"]] == [
        "request_to_block_assignment"
    ]


def test_live_request_contract_rejects_unbounded_workload() -> None:
    script = _load_script_module()

    try:
        script.capture_live_request_contract_observation(
            output_dir=Path("/tmp/cachepawl-live-request-test"),
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
            "name": "live_request_id",
            "status": "observed",
            "evidence": {"request_id": "0"},
            "blocker_reason": None,
        },
        {
            "name": "request_to_block_assignment",
            "status": "observed" if not field_blockers else "blocked",
            "evidence": {"active_sample_block_ids": (3, 4)},
            "blocker_reason": None if not field_blockers else "no block ids",
        },
        {
            "name": "scheduler_request_metadata",
            "status": "observed",
            "evidence": {"phases_with_request": ("after_enqueue_before_step",)},
            "blocker_reason": None,
        },
        {
            "name": "before_after_block_pool_usage",
            "status": "observed",
            "evidence": {"usage_changed": False},
            "blocker_reason": None,
        },
    ]
    return {
        "status": "live_request_contract_observation",
        "runtime_path": "LLM.llm_engine.engine_core.engine_core",
        "request_id": "0",
        "prompt": "Count from one to four:",
        "max_new_tokens": 8,
        "snapshots": [
            {
                "phase": "before_enqueue",
                "block_usage": {"block_pool_free_blocks": 10},
            },
            {
                "phase": "after_first_step",
                "request_block_ids": {"total_block_id_count": 2, "sample_block_ids": (3, 4)},
            },
            {
                "phase": "after_completion",
                "block_usage": {"block_pool_free_blocks": 10},
            },
        ],
        "output_metadata": {"completion_request_ids": ("0",)},
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
        },
    }


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "capture_vllm_live_request_contract_observation_for_test",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
