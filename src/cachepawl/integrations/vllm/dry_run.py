"""Planner-level dry-run probes for translated vLLM cache plans.

The dry-run result is advisory only: it computes a proposed Cachepawl planner
view beside vanilla vLLM planning output and never returns that proposal to vLLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from cachepawl.integrations.vllm.advisory import (
    MISSING_FIELDS_FOR_MUTATION,
    JsonLike,
    VllmCacheAdvisoryReport,
    advise_vllm_runtime_cache_plan,
)


@dataclass(frozen=True, slots=True)
class VllmDryRunGroupProposal:
    """Per-group proposed planner view for the dry-run result."""

    group_index: int
    cache_kind: str | None
    layer_count: int
    vanilla_estimated_bytes: int | None
    cachepawl_proposed_bytes: int | None
    proposed_savings_bytes: int | None
    block_size: int | None
    page_size_bytes: int | None
    useful_bytes: int | None

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "group_index": self.group_index,
            "cache_kind": self.cache_kind,
            "layer_count": self.layer_count,
            "vanilla_estimated_bytes": self.vanilla_estimated_bytes,
            "cachepawl_proposed_bytes": self.cachepawl_proposed_bytes,
            "proposed_savings_bytes": self.proposed_savings_bytes,
            "block_size": self.block_size,
            "page_size_bytes": self.page_size_bytes,
            "useful_bytes": self.useful_bytes,
        }


@dataclass(frozen=True, slots=True)
class VllmPlannerDryRunResult:
    """Serializable planner-level dry-run result."""

    status: str
    safe_for_advisory_only: bool
    returned_to_vllm: bool
    vllm_behavior_changed: bool
    vanilla_observed_reserved_bytes: int | None
    vanilla_observed_useful_bytes: int | None
    cachepawl_proposed_reserved_bytes: int | None
    estimated_savings_bytes: int | None
    overestimation_ratio: float | None
    wasted_fraction: float | None
    planner_input_coverage: dict[str, bool] = field(default_factory=dict)
    missing_fields_that_prevent_mutation: tuple[str, ...] = MISSING_FIELDS_FOR_MUTATION
    group_proposals: tuple[VllmDryRunGroupProposal, ...] = ()

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "status": self.status,
            "safe_for_advisory_only": self.safe_for_advisory_only,
            "returned_to_vllm": self.returned_to_vllm,
            "vllm_behavior_changed": self.vllm_behavior_changed,
            "vanilla_observed_reserved_bytes": self.vanilla_observed_reserved_bytes,
            "vanilla_observed_useful_bytes": self.vanilla_observed_useful_bytes,
            "cachepawl_proposed_reserved_bytes": self.cachepawl_proposed_reserved_bytes,
            "estimated_savings_bytes": self.estimated_savings_bytes,
            "overestimation_ratio": self.overestimation_ratio,
            "wasted_fraction": self.wasted_fraction,
            "planner_input_coverage": dict(sorted(self.planner_input_coverage.items())),
            "missing_fields_that_prevent_mutation": self.missing_fields_that_prevent_mutation,
            "group_proposals": tuple(group.to_dict() for group in self.group_proposals),
        }


RuntimeObservationLike: TypeAlias = object


def dry_run_vllm_planner_probe(
    translated_or_observation: RuntimeObservationLike,
    *,
    raw_safe_metadata: object | None = None,
) -> VllmPlannerDryRunResult:
    """Compute a Cachepawl proposed planner view without mutating vLLM."""

    translated, metadata = _translated_and_metadata(
        translated_or_observation,
        explicit_metadata=raw_safe_metadata,
    )
    advisory = advise_vllm_runtime_cache_plan(
        translated,
        raw_safe_metadata=metadata,
    )
    return _dry_run_from_advisory(advisory)


def _dry_run_from_advisory(advisory: VllmCacheAdvisoryReport) -> VllmPlannerDryRunResult:
    status = (
        "planner_dry_run_available"
        if advisory.primary_classification == "planner_advisory_available"
        else "insufficient_data"
    )
    return VllmPlannerDryRunResult(
        status=status,
        safe_for_advisory_only=True,
        returned_to_vllm=False,
        vllm_behavior_changed=False,
        vanilla_observed_reserved_bytes=advisory.observed_reserved_bytes,
        vanilla_observed_useful_bytes=advisory.observed_useful_bytes,
        cachepawl_proposed_reserved_bytes=advisory.cachepawl_recommended_bytes,
        estimated_savings_bytes=advisory.advisory_savings_bytes,
        overestimation_ratio=advisory.overestimation_ratio,
        wasted_fraction=advisory.wasted_fraction,
        planner_input_coverage=advisory.planner_input_coverage,
        group_proposals=tuple(
            VllmDryRunGroupProposal(
                group_index=group.group_index,
                cache_kind=group.cache_kind,
                layer_count=group.layer_count,
                vanilla_estimated_bytes=group.estimated_bytes,
                cachepawl_proposed_bytes=(
                    group.useful_bytes * group.layer_count * advisory.num_blocks
                    if group.useful_bytes is not None and advisory.num_blocks is not None
                    else None
                ),
                proposed_savings_bytes=(
                    group.estimated_bytes
                    - (group.useful_bytes * group.layer_count * advisory.num_blocks)
                    if (
                        group.estimated_bytes is not None
                        and group.useful_bytes is not None
                        and advisory.num_blocks is not None
                    )
                    else None
                ),
                block_size=group.block_size,
                page_size_bytes=group.page_size_bytes,
                useful_bytes=group.useful_bytes,
            )
            for group in advisory.group_advisories
        ),
    )


def _translated_and_metadata(
    value: object,
    *,
    explicit_metadata: object | None,
) -> tuple[object, object | None]:
    if isinstance(value, dict) and "translated_runtime_cache_config" in value:
        return (
            value["translated_runtime_cache_config"],
            explicit_metadata if explicit_metadata is not None else value.get("raw_safe_metadata"),
        )
    return value, explicit_metadata
