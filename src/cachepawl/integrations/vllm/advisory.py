"""Advisory diagnostics for translated vLLM runtime cache plans.

This module is import-safe without vLLM. It consumes Cachepawl-owned translated
runtime snapshots and emits read-only metrics for observer-in-the-loop planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

JsonLike: TypeAlias = object
VllmCacheAdvisoryClassification: TypeAlias = Literal[
    "observe_only",
    "planner_advisory_available",
    "mutation_required_for_runtime_effect",
    "insufficient_data",
]

MISSING_FIELDS_FOR_MUTATION: tuple[str, ...] = (
    "stable scheduler or planner construction hook",
    "allocator or KVCacheManager replacement control point",
    "worker tensor allocation layout control point",
    "runtime request-to-block assignment control",
    "Mamba state-index and attention view rewrite contract",
)


@dataclass(frozen=True, slots=True)
class VllmCacheGroupAdvisory:
    """Advisory metrics for one translated vLLM cache group."""

    group_index: int
    cache_kind: str | None
    layer_count: int
    block_size: int | None
    page_size_bytes: int | None
    useful_bytes: int | None
    estimated_bytes: int | None
    overestimation_ratio: float | None
    wasted_fraction: float | None
    missing_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "group_index": self.group_index,
            "cache_kind": self.cache_kind,
            "layer_count": self.layer_count,
            "block_size": self.block_size,
            "page_size_bytes": self.page_size_bytes,
            "useful_bytes": self.useful_bytes,
            "estimated_bytes": self.estimated_bytes,
            "overestimation_ratio": self.overestimation_ratio,
            "wasted_fraction": self.wasted_fraction,
            "missing_fields": self.missing_fields,
        }


@dataclass(frozen=True, slots=True)
class VllmCacheAdvisoryReport:
    """Serializable advisory report for a translated runtime cache config."""

    primary_classification: VllmCacheAdvisoryClassification
    classifications: tuple[VllmCacheAdvisoryClassification, ...]
    num_blocks: int | None
    cache_group_count: int
    cache_tensor_count: int
    layer_count: int
    available_kv_cache_gpu_memory_bytes: int | None
    observed_reserved_bytes: int | None
    observed_useful_bytes: int | None
    cachepawl_recommended_bytes: int | None
    advisory_savings_bytes: int | None
    overestimation_ratio: float | None
    wasted_fraction: float | None
    planner_input_coverage: dict[str, bool] = field(default_factory=dict)
    missing_fields_for_mutation: tuple[str, ...] = MISSING_FIELDS_FOR_MUTATION
    group_advisories: tuple[VllmCacheGroupAdvisory, ...] = ()

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "primary_classification": self.primary_classification,
            "classifications": self.classifications,
            "num_blocks": self.num_blocks,
            "cache_group_count": self.cache_group_count,
            "cache_tensor_count": self.cache_tensor_count,
            "layer_count": self.layer_count,
            "available_kv_cache_gpu_memory_bytes": self.available_kv_cache_gpu_memory_bytes,
            "observed_reserved_bytes": self.observed_reserved_bytes,
            "observed_useful_bytes": self.observed_useful_bytes,
            "cachepawl_recommended_bytes": self.cachepawl_recommended_bytes,
            "advisory_savings_bytes": self.advisory_savings_bytes,
            "overestimation_ratio": self.overestimation_ratio,
            "wasted_fraction": self.wasted_fraction,
            "planner_input_coverage": dict(sorted(self.planner_input_coverage.items())),
            "missing_fields_for_mutation": self.missing_fields_for_mutation,
            "group_advisories": tuple(group.to_dict() for group in self.group_advisories),
        }


def advise_vllm_runtime_cache_plan(
    translated_cache_config: object,
    *,
    raw_safe_metadata: object | None = None,
) -> VllmCacheAdvisoryReport:
    """Compute read-only advisory metrics from translated vLLM runtime output."""

    config = _as_mapping(translated_cache_config, "translated_cache_config")
    metadata = (
        _as_mapping(raw_safe_metadata, "raw_safe_metadata") if raw_safe_metadata is not None else {}
    )
    groups = _as_sequence(config.get("groups"), "groups")
    tensors = _as_sequence(config.get("tensors"), "tensors")
    num_blocks = _optional_int(config.get("num_blocks"), "num_blocks")
    group_advisories = tuple(_group_advisory(group, num_blocks) for group in groups)

    observed_reserved_bytes = _observed_reserved_bytes(tensors, group_advisories, num_blocks)
    observed_useful_bytes = _observed_useful_bytes(config, group_advisories, num_blocks)
    cachepawl_recommended_bytes = observed_useful_bytes
    advisory_savings_bytes = (
        observed_reserved_bytes - cachepawl_recommended_bytes
        if observed_reserved_bytes is not None and cachepawl_recommended_bytes is not None
        else None
    )
    classifications = _classifications(
        num_blocks=num_blocks,
        group_advisories=group_advisories,
        observed_reserved_bytes=observed_reserved_bytes,
        observed_useful_bytes=observed_useful_bytes,
    )

    return VllmCacheAdvisoryReport(
        primary_classification=(
            "planner_advisory_available"
            if "planner_advisory_available" in classifications
            else "insufficient_data"
        ),
        classifications=classifications,
        num_blocks=num_blocks,
        cache_group_count=_optional_int(config.get("group_count"), "group_count") or len(groups),
        cache_tensor_count=len(tensors),
        layer_count=_optional_int(config.get("layer_count"), "layer_count") or _sum_layers(groups),
        available_kv_cache_gpu_memory_bytes=_optional_int(
            metadata.get("available_gpu_memory_for_kv_cache"),
            "available_gpu_memory_for_kv_cache",
        ),
        observed_reserved_bytes=observed_reserved_bytes,
        observed_useful_bytes=observed_useful_bytes,
        cachepawl_recommended_bytes=cachepawl_recommended_bytes,
        advisory_savings_bytes=advisory_savings_bytes,
        overestimation_ratio=_ratio(observed_reserved_bytes, observed_useful_bytes),
        wasted_fraction=_wasted_fraction(observed_reserved_bytes, observed_useful_bytes),
        planner_input_coverage=_planner_input_coverage(
            config=config,
            metadata=metadata,
            groups=groups,
            tensors=tensors,
            group_advisories=group_advisories,
        ),
        group_advisories=group_advisories,
    )


def _group_advisory(group: object, num_blocks: int | None) -> VllmCacheGroupAdvisory:
    mapping = _as_mapping(group, "group")
    cache_spec = _as_mapping(mapping.get("cache_spec"), "group.cache_spec")
    layer_count = _optional_int(mapping.get("layer_count"), "group.layer_count") or len(
        _as_sequence(mapping.get("layer_names"), "group.layer_names")
    )
    page_size_bytes = _optional_int(cache_spec.get("page_size_bytes"), "page_size_bytes")
    useful_bytes = _optional_int(cache_spec.get("useful_bytes"), "useful_bytes")
    estimated_bytes = (
        page_size_bytes * num_blocks * layer_count
        if page_size_bytes is not None and num_blocks is not None
        else None
    )
    useful_total = (
        useful_bytes * num_blocks * layer_count
        if useful_bytes is not None and num_blocks is not None
        else None
    )
    missing_fields = []
    if page_size_bytes is None:
        missing_fields.append("page_size_bytes")
    if useful_bytes is None:
        missing_fields.append("useful_bytes")
    if num_blocks is None:
        missing_fields.append("num_blocks")

    return VllmCacheGroupAdvisory(
        group_index=_optional_int(mapping.get("group_index"), "group_index") or 0,
        cache_kind=_optional_str(cache_spec.get("cache_kind")),
        layer_count=layer_count,
        block_size=_optional_int(cache_spec.get("block_size"), "block_size"),
        page_size_bytes=page_size_bytes,
        useful_bytes=useful_bytes,
        estimated_bytes=estimated_bytes,
        overestimation_ratio=_ratio(estimated_bytes, useful_total),
        wasted_fraction=_wasted_fraction(estimated_bytes, useful_total),
        missing_fields=tuple(missing_fields),
    )


def _observed_reserved_bytes(
    tensors: tuple[object, ...],
    group_advisories: tuple[VllmCacheGroupAdvisory, ...],
    num_blocks: int | None,
) -> int | None:
    tensor_total = 0
    for tensor in tensors:
        size = _optional_int(_as_mapping(tensor, "tensor").get("size_bytes"), "size_bytes")
        if size is None:
            return None
        tensor_total += size
    if tensors:
        return tensor_total
    if num_blocks is None:
        return None
    group_totals = [group.estimated_bytes for group in group_advisories]
    if any(total is None for total in group_totals):
        return None
    return sum(cast(int, total) for total in group_totals)


def _observed_useful_bytes(
    config: dict[str, object],
    group_advisories: tuple[VllmCacheGroupAdvisory, ...],
    num_blocks: int | None,
) -> int | None:
    total_useful_page = _optional_int(config.get("total_useful_bytes"), "total_useful_bytes")
    if total_useful_page is not None and num_blocks is not None:
        return total_useful_page * num_blocks
    if num_blocks is None:
        return None
    total = 0
    for group in group_advisories:
        if group.useful_bytes is None:
            return None
        total += group.useful_bytes * group.layer_count * num_blocks
    return total


def _classifications(
    *,
    num_blocks: int | None,
    group_advisories: tuple[VllmCacheGroupAdvisory, ...],
    observed_reserved_bytes: int | None,
    observed_useful_bytes: int | None,
) -> tuple[VllmCacheAdvisoryClassification, ...]:
    if (
        num_blocks is None
        or not group_advisories
        or observed_reserved_bytes is None
        or observed_useful_bytes is None
    ):
        return ("observe_only", "insufficient_data", "mutation_required_for_runtime_effect")
    return (
        "observe_only",
        "planner_advisory_available",
        "mutation_required_for_runtime_effect",
    )


def _planner_input_coverage(
    *,
    config: dict[str, object],
    metadata: dict[str, object],
    groups: tuple[object, ...],
    tensors: tuple[object, ...],
    group_advisories: tuple[VllmCacheGroupAdvisory, ...],
) -> dict[str, bool]:
    return {
        "num_blocks": _optional_int(config.get("num_blocks"), "num_blocks") is not None,
        "cache_group_count": bool(groups),
        "cache_tensor_count": bool(tensors),
        "layer_count": _optional_int(config.get("layer_count"), "layer_count") is not None,
        "per_group_cache_kind": all(group.cache_kind is not None for group in group_advisories),
        "per_group_block_size": all(group.block_size is not None for group in group_advisories),
        "per_group_page_size_bytes": all(
            group.page_size_bytes is not None for group in group_advisories
        ),
        "per_group_useful_bytes": all(group.useful_bytes is not None for group in group_advisories),
        "available_kv_cache_gpu_memory": _optional_int(
            metadata.get("available_gpu_memory_for_kv_cache"),
            "available_gpu_memory_for_kv_cache",
        )
        is not None,
    }


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


def _sum_layers(groups: tuple[object, ...]) -> int:
    total = 0
    for group in groups:
        mapping = _as_mapping(group, "group")
        total += _optional_int(mapping.get("layer_count"), "group.layer_count") or len(
            _as_sequence(mapping.get("layer_names"), "group.layer_names")
        )
    return total


def _as_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _as_sequence(value: object, name: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise TypeError(f"{name} must be a sequence")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc


def _optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer when provided")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
