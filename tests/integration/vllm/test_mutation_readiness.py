"""Tests for vLLM mutation-readiness artifact checks."""

from __future__ import annotations

import copy

import cachepawl.integrations.vllm as vllm_integration
from cachepawl.integrations.vllm import check_vllm_mutation_readiness


def test_mutation_readiness_is_exported_without_vllm_dependency() -> None:
    assert "check_vllm_mutation_readiness" in vllm_integration.__all__


def test_readiness_recommends_advisory_only_when_mutation_fields_are_missing() -> None:
    report = check_vllm_mutation_readiness(
        _planner_fixture(),
        _diff_fixture(),
        group_level_diff=_group_diff_fixture(),
    )

    assert report.classification == "advisory_only_recommended"
    assert report.ready_for_controlled_substitution is False
    assert report.advisory_only is True
    assert report.non_mutating is True
    assert report.returned_to_vllm is False
    assert report.vllm_behavior_changed is False
    assert report.failed_invariants == ()
    assert report.blocked_invariants == ("mutation_required_missing_fields",)
    assert "num_blocks_compatibility" in report.passed_invariants
    assert report.mutation_required_missing_fields == ("stable opt-in substitution control point",)


def test_readiness_blocks_when_structural_invariant_fails() -> None:
    diff = _diff_fixture()
    diff["num_blocks"] = 11

    report = check_vllm_mutation_readiness(
        _planner_fixture(),
        diff,
        group_level_diff=_group_diff_fixture(),
    )

    assert report.classification == "blocked_missing_invariants"
    assert "num_blocks_compatibility" in report.failed_invariants
    assert "mutation_required_missing_fields" in report.blocked_invariants


def test_readiness_can_classify_controlled_substitution_ready() -> None:
    diff = _diff_fixture()
    diff["missing_fields_that_prevent_mutation"] = []

    report = check_vllm_mutation_readiness(
        _planner_fixture(),
        diff,
        group_level_diff=_group_diff_fixture(),
    )

    assert report.classification == "ready_for_controlled_substitution"
    assert report.ready_for_controlled_substitution is True
    assert report.failed_invariants == ()
    assert report.blocked_invariants == ()


def _planner_fixture() -> dict[str, object]:
    return {
        "num_blocks": 10,
        "group_count": 2,
        "layer_count": 3,
        "attention_group_count": 1,
        "mamba_group_count": 1,
        "groups": [
            {
                "group_index": 0,
                "layer_count": 1,
                "layer_names": ["attn.0"],
                "cache_spec": {
                    "cache_kind": "attention",
                    "spec_type": "FullAttentionSpec",
                    "dtype": "torch.bfloat16",
                    "block_size": 16,
                    "page_size_bytes": 100,
                    "useful_bytes": 100,
                },
            },
            {
                "group_index": 1,
                "layer_count": 2,
                "layer_names": ["mamba.0", "mamba.1"],
                "cache_spec": {
                    "cache_kind": "mamba",
                    "spec_type": "MambaSpec",
                    "dtype": None,
                    "block_size": 4096,
                    "page_size_bytes": 100,
                    "useful_bytes": 60,
                    "metadata": {
                        "dtypes": ["torch.bfloat16", "torch.bfloat16"],
                        "shapes": [[3, 8], [2, 4, 4]],
                    },
                },
            },
        ],
        "tensors": [
            {"size_bytes": 1000, "shared_by": ["attn.0"]},
            {"size_bytes": 1000, "shared_by": ["mamba.0"]},
            {"size_bytes": 1000, "shared_by": ["mamba.1"]},
        ],
        "total_useful_bytes": 220,
    }


def _diff_fixture() -> dict[str, object]:
    return {
        "num_blocks": 10,
        "cache_group_count": 2,
        "cache_tensor_count": 3,
        "layer_count": 3,
        "vanilla_reserved_bytes": 3000,
        "vanilla_useful_bytes": 2200,
        "cachepawl_proposed_reserved_bytes": 2200,
        "estimated_savings_bytes": 800,
        "missing_fields_that_prevent_mutation": [
            "stable opt-in substitution control point",
        ],
    }


def _group_diff_fixture() -> list[dict[str, object]]:
    return copy.deepcopy(
        [
            {
                "group_index": 0,
                "block_size": 16,
                "page_size_bytes": 100,
                "useful_bytes": 100,
                "vanilla_reserved_bytes": 1000,
                "cachepawl_proposed_reserved_bytes": 1000,
                "estimated_savings_bytes": 0,
                "overestimation_ratio": 1.0,
                "wasted_fraction": 0.0,
            },
            {
                "group_index": 1,
                "block_size": 4096,
                "page_size_bytes": 100,
                "useful_bytes": 60,
                "vanilla_reserved_bytes": 2000,
                "cachepawl_proposed_reserved_bytes": 1200,
                "estimated_savings_bytes": 800,
                "overestimation_ratio": 2000 / 1200,
                "wasted_fraction": 0.4,
            },
        ]
    )
