"""Tests for vLLM planner-stage advisory diffs."""

from __future__ import annotations

import json

import cachepawl.integrations.vllm as vllm_integration
from cachepawl.integrations.vllm import diff_vllm_planner_stage_advisory


def test_package_exports_planner_diff_without_vllm_dependency() -> None:
    assert "diff_vllm_planner_stage_advisory" in vllm_integration.__all__


def test_planner_stage_diff_computes_non_mutating_metrics() -> None:
    translated = {
        "num_blocks": 10,
        "group_count": 2,
        "layer_count": 3,
        "groups": (
            {
                "group_index": 0,
                "layer_count": 1,
                "layer_names": ("attn.0",),
                "cache_spec": {
                    "cache_kind": "attention",
                    "block_size": 16,
                    "page_size_bytes": 100,
                    "useful_bytes": 100,
                },
            },
            {
                "group_index": 1,
                "layer_count": 2,
                "layer_names": ("mamba.0", "mamba.1"),
                "cache_spec": {
                    "cache_kind": "mamba",
                    "block_size": 4096,
                    "page_size_bytes": 100,
                    "useful_bytes": 60,
                },
            },
        ),
        "tensors": (
            {"size_bytes": 1000, "shared_by": ("attn.0",)},
            {"size_bytes": 1000, "shared_by": ("mamba.0",)},
            {"size_bytes": 1000, "shared_by": ("mamba.1",)},
        ),
        "total_useful_bytes": 220,
    }

    result = diff_vllm_planner_stage_advisory(
        translated,
        raw_safe_metadata={
            "available_memory": [4096],
            "planner_matches_runtime_scheduler": True,
            "runtime_changed_during_replay": False,
        },
    )
    payload = result.to_dict()

    assert result.status == "planner_stage_advisory_diff_available"
    assert result.classification == "planner_advisory_available"
    assert result.non_mutating is True
    assert result.returned_to_vllm is False
    assert result.vllm_behavior_changed is False
    assert result.vanilla_reserved_bytes == 3000
    assert result.vanilla_useful_bytes == 2200
    assert result.cachepawl_proposed_reserved_bytes == 2200
    assert result.estimated_savings_bytes == 800
    assert result.overestimation_ratio == 3000 / 2200
    assert result.wasted_fraction == (3000 - 2200) / 3000
    assert result.cache_group_count == 2
    assert result.cache_tensor_count == 3
    assert result.layer_count == 3
    assert result.num_blocks == 10
    assert result.parity_status["planner_matches_runtime_scheduler"] is True
    assert result.parity_status["runtime_changed_during_replay"] is False
    assert result.planner_input_coverage["available_kv_cache_gpu_memory"] is True
    assert result.group_level_diff[1].estimated_savings_bytes == 800
    assert result.group_level_diff[1].wasted_fraction == 0.4
    assert "stable scheduler or planner construction hook" in (
        result.missing_fields_that_prevent_mutation
    )
    json.dumps(payload, sort_keys=True)


def test_planner_stage_diff_reports_missing_fields() -> None:
    translated = {
        "num_blocks": 10,
        "groups": (
            {
                "group_index": 0,
                "layer_count": 1,
                "cache_spec": {
                    "cache_kind": "attention",
                    "block_size": 16,
                    "page_size_bytes": 100,
                    "useful_bytes": None,
                },
            },
        ),
        "tensors": ({"size_bytes": 1000, "shared_by": ("attn.0",)},),
    }

    result = diff_vllm_planner_stage_advisory(translated)

    assert result.status == "insufficient_data"
    assert result.vanilla_reserved_bytes == 1000
    assert result.vanilla_useful_bytes is None
    assert result.cachepawl_proposed_reserved_bytes is None
    assert result.parity_status["source"] == "not_provided"
