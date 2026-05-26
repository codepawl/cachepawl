"""Planner-stage advisory diffs for translated vLLM planner output.

This module is import-safe without vLLM. It consumes Cachepawl-owned translated
planner-stage artifacts and computes a sidecar advisory diff without returning
anything to vLLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from cachepawl.integrations.vllm.advisory import (
    JsonLike,
    advise_vllm_runtime_cache_plan,
)
from cachepawl.integrations.vllm.dry_run import dry_run_vllm_planner_probe

JsonObject: TypeAlias = dict[str, object]


@dataclass(frozen=True, slots=True)
class VllmPlannerStageGroupDiff:
    """Per-group planner-stage advisory diff."""

    group_index: int
    cache_kind: str | None
    layer_count: int
    vanilla_reserved_bytes: int | None
    cachepawl_proposed_reserved_bytes: int | None
    estimated_savings_bytes: int | None
    block_size: int | None
    page_size_bytes: int | None
    useful_bytes: int | None
    overestimation_ratio: float | None
    wasted_fraction: float | None

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "group_index": self.group_index,
            "cache_kind": self.cache_kind,
            "layer_count": self.layer_count,
            "vanilla_reserved_bytes": self.vanilla_reserved_bytes,
            "cachepawl_proposed_reserved_bytes": self.cachepawl_proposed_reserved_bytes,
            "estimated_savings_bytes": self.estimated_savings_bytes,
            "block_size": self.block_size,
            "page_size_bytes": self.page_size_bytes,
            "useful_bytes": self.useful_bytes,
            "overestimation_ratio": self.overestimation_ratio,
            "wasted_fraction": self.wasted_fraction,
        }


@dataclass(frozen=True, slots=True)
class VllmPlannerStageAdvisoryDiff:
    """Serializable planner-stage post-call advisory diff."""

    status: str
    classification: str
    advisory_only: bool
    non_mutating: bool
    returned_to_vllm: bool
    vllm_behavior_changed: bool
    vanilla_reserved_bytes: int | None
    vanilla_useful_bytes: int | None
    cachepawl_proposed_reserved_bytes: int | None
    estimated_savings_bytes: int | None
    overestimation_ratio: float | None
    wasted_fraction: float | None
    cache_group_count: int
    cache_tensor_count: int
    layer_count: int
    num_blocks: int | None
    parity_status: dict[str, JsonLike]
    missing_fields_that_prevent_mutation: tuple[str, ...]
    planner_input_coverage: dict[str, bool]
    group_level_diff: tuple[VllmPlannerStageGroupDiff, ...]

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "status": self.status,
            "classification": self.classification,
            "advisory_only": self.advisory_only,
            "non_mutating": self.non_mutating,
            "returned_to_vllm": self.returned_to_vllm,
            "vllm_behavior_changed": self.vllm_behavior_changed,
            "vanilla_reserved_bytes": self.vanilla_reserved_bytes,
            "vanilla_useful_bytes": self.vanilla_useful_bytes,
            "cachepawl_proposed_reserved_bytes": self.cachepawl_proposed_reserved_bytes,
            "estimated_savings_bytes": self.estimated_savings_bytes,
            "overestimation_ratio": self.overestimation_ratio,
            "wasted_fraction": self.wasted_fraction,
            "cache_group_count": self.cache_group_count,
            "cache_tensor_count": self.cache_tensor_count,
            "layer_count": self.layer_count,
            "num_blocks": self.num_blocks,
            "parity_status": self.parity_status,
            "missing_fields_that_prevent_mutation": self.missing_fields_that_prevent_mutation,
            "planner_input_coverage": dict(sorted(self.planner_input_coverage.items())),
            "group_level_diff": tuple(group.to_dict() for group in self.group_level_diff),
        }


def diff_vllm_planner_stage_advisory(
    translated_planner_stage_config: object,
    *,
    raw_safe_metadata: object | None = None,
) -> VllmPlannerStageAdvisoryDiff:
    """Compute a non-mutating advisory diff from translated planner output."""

    advisory_metadata = _metadata_for_advisory(raw_safe_metadata)
    advisory = advise_vllm_runtime_cache_plan(
        translated_planner_stage_config,
        raw_safe_metadata=advisory_metadata,
    )
    dry_run = dry_run_vllm_planner_probe(
        translated_planner_stage_config,
        raw_safe_metadata=advisory_metadata,
    )
    group_level_diff = tuple(
        VllmPlannerStageGroupDiff(
            group_index=group.group_index,
            cache_kind=group.cache_kind,
            layer_count=group.layer_count,
            vanilla_reserved_bytes=group.vanilla_estimated_bytes,
            cachepawl_proposed_reserved_bytes=group.cachepawl_proposed_bytes,
            estimated_savings_bytes=group.proposed_savings_bytes,
            block_size=group.block_size,
            page_size_bytes=group.page_size_bytes,
            useful_bytes=group.useful_bytes,
            overestimation_ratio=_ratio(
                group.vanilla_estimated_bytes,
                group.cachepawl_proposed_bytes,
            ),
            wasted_fraction=_wasted_fraction(
                group.vanilla_estimated_bytes,
                group.cachepawl_proposed_bytes,
            ),
        )
        for group in dry_run.group_proposals
    )
    status = (
        "planner_stage_advisory_diff_available"
        if dry_run.status == "planner_dry_run_available"
        else "insufficient_data"
    )
    return VllmPlannerStageAdvisoryDiff(
        status=status,
        classification=advisory.primary_classification,
        advisory_only=True,
        non_mutating=True,
        returned_to_vllm=False,
        vllm_behavior_changed=False,
        vanilla_reserved_bytes=dry_run.vanilla_observed_reserved_bytes,
        vanilla_useful_bytes=dry_run.vanilla_observed_useful_bytes,
        cachepawl_proposed_reserved_bytes=dry_run.cachepawl_proposed_reserved_bytes,
        estimated_savings_bytes=dry_run.estimated_savings_bytes,
        overestimation_ratio=dry_run.overestimation_ratio,
        wasted_fraction=dry_run.wasted_fraction,
        cache_group_count=advisory.cache_group_count,
        cache_tensor_count=advisory.cache_tensor_count,
        layer_count=advisory.layer_count,
        num_blocks=advisory.num_blocks,
        parity_status=_parity_status(raw_safe_metadata),
        missing_fields_that_prevent_mutation=dry_run.missing_fields_that_prevent_mutation,
        planner_input_coverage=dry_run.planner_input_coverage,
        group_level_diff=group_level_diff,
    )


def _parity_status(raw_safe_metadata: object | None) -> dict[str, JsonLike]:
    metadata = raw_safe_metadata if isinstance(raw_safe_metadata, dict) else {}
    return {
        "planner_matches_runtime_scheduler": _optional_bool(
            metadata.get("planner_matches_runtime_scheduler")
        ),
        "runtime_changed_during_replay": _optional_bool(
            metadata.get("runtime_changed_during_replay")
        ),
        "source": "raw_safe_metadata" if metadata else "not_provided",
        "non_mutating": True,
        "returned_to_vllm": False,
        "vllm_behavior_changed": False,
    }


def _metadata_for_advisory(raw_safe_metadata: object | None) -> object | None:
    if not isinstance(raw_safe_metadata, dict):
        return raw_safe_metadata
    metadata = dict(raw_safe_metadata)
    if "available_gpu_memory_for_kv_cache" not in metadata:
        available_memory = metadata.get("available_memory")
        if isinstance(available_memory, list | tuple) and available_memory:
            metadata["available_gpu_memory_for_kv_cache"] = available_memory[0]
    return metadata


def _ratio(estimated_bytes: int | None, useful_bytes: int | None) -> float | None:
    if estimated_bytes is None or useful_bytes is None:
        return None
    if useful_bytes == 0:
        return 0.0 if estimated_bytes == 0 else None
    return estimated_bytes / useful_bytes


def _wasted_fraction(estimated_bytes: int | None, useful_bytes: int | None) -> float | None:
    if estimated_bytes is None or useful_bytes is None:
        return None
    if estimated_bytes == 0:
        return 0.0
    return min(1.0, max(0.0, (estimated_bytes - useful_bytes) / estimated_bytes))


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None
