"""Tests for vLLM runtime cache-plan advisory diagnostics."""

from __future__ import annotations

import json

import cachepawl.integrations.vllm as vllm_integration
from cachepawl.integrations.vllm import advise_vllm_runtime_cache_plan


def test_package_exports_advisory_without_vllm_dependency() -> None:
    assert "advise_vllm_runtime_cache_plan" in vllm_integration.__all__


def test_advisory_report_computes_runtime_planner_metrics() -> None:
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
    report = advise_vllm_runtime_cache_plan(
        translated,
        raw_safe_metadata={"available_gpu_memory_for_kv_cache": 4096},
    )
    payload = report.to_dict()

    assert report.primary_classification == "planner_advisory_available"
    assert report.classifications == (
        "observe_only",
        "planner_advisory_available",
        "mutation_required_for_runtime_effect",
    )
    assert report.observed_reserved_bytes == 3000
    assert report.observed_useful_bytes == 2200
    assert report.cachepawl_recommended_bytes == 2200
    assert report.advisory_savings_bytes == 800
    assert report.overestimation_ratio == 3000 / 2200
    assert report.wasted_fraction == (3000 - 2200) / 3000
    assert report.available_kv_cache_gpu_memory_bytes == 4096
    assert payload["planner_input_coverage"] == {
        "available_kv_cache_gpu_memory": True,
        "cache_group_count": True,
        "cache_tensor_count": True,
        "layer_count": True,
        "num_blocks": True,
        "per_group_block_size": True,
        "per_group_cache_kind": True,
        "per_group_page_size_bytes": True,
        "per_group_useful_bytes": True,
    }
    assert report.group_advisories[1].wasted_fraction == 0.4
    json.dumps(payload, sort_keys=True)


def test_advisory_report_marks_insufficient_data_when_useful_bytes_missing() -> None:
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

    report = advise_vllm_runtime_cache_plan(translated)

    assert report.primary_classification == "insufficient_data"
    assert report.classifications == (
        "observe_only",
        "insufficient_data",
        "mutation_required_for_runtime_effect",
    )
    assert report.observed_reserved_bytes == 1000
    assert report.observed_useful_bytes is None
    assert "worker tensor allocation layout control point" in report.missing_fields_for_mutation
