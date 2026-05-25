"""Tests for vLLM planner dry-run probes."""

from __future__ import annotations

import json

import cachepawl.integrations.vllm as vllm_integration
from cachepawl.integrations.vllm import dry_run_vllm_planner_probe


def test_package_exports_dry_run_without_vllm_dependency() -> None:
    assert "dry_run_vllm_planner_probe" in vllm_integration.__all__


def test_dry_run_computes_proposed_plan_from_translated_config() -> None:
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

    result = dry_run_vllm_planner_probe(translated)
    payload = result.to_dict()

    assert result.status == "planner_dry_run_available"
    assert result.safe_for_advisory_only is True
    assert result.returned_to_vllm is False
    assert result.vllm_behavior_changed is False
    assert result.vanilla_observed_reserved_bytes == 3000
    assert result.vanilla_observed_useful_bytes == 2200
    assert result.cachepawl_proposed_reserved_bytes == 2200
    assert result.estimated_savings_bytes == 800
    assert result.overestimation_ratio == 3000 / 2200
    assert result.wasted_fraction == (3000 - 2200) / 3000
    assert result.group_proposals[1].cachepawl_proposed_bytes == 1200
    assert result.group_proposals[1].proposed_savings_bytes == 800
    assert "stable scheduler or planner construction hook" in (
        result.missing_fields_that_prevent_mutation
    )
    json.dumps(payload, sort_keys=True)


def test_dry_run_accepts_runtime_observation_payload_shape() -> None:
    observation = {
        "status": "runtime_resolved_translation",
        "translated_runtime_cache_config": {
            "num_blocks": 10,
            "groups": (
                {
                    "group_index": 0,
                    "layer_count": 1,
                    "cache_spec": {
                        "cache_kind": "attention",
                        "block_size": 16,
                        "page_size_bytes": 100,
                        "useful_bytes": 80,
                    },
                },
            ),
            "tensors": ({"size_bytes": 1000, "shared_by": ("attn.0",)},),
            "total_useful_bytes": 80,
        },
        "raw_safe_metadata": {"available_gpu_memory_for_kv_cache": 4096},
    }

    result = dry_run_vllm_planner_probe(observation)

    assert result.status == "planner_dry_run_available"
    assert result.vanilla_observed_reserved_bytes == 1000
    assert result.vanilla_observed_useful_bytes == 800
    assert result.cachepawl_proposed_reserved_bytes == 800
