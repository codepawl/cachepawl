"""Mutation-readiness checks for translated vLLM planner artifacts.

This module is import-safe without vLLM. It validates serialized Cachepawl
artifacts and reports whether they contain enough evidence for a future,
explicitly opt-in mutation experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Literal, TypeAlias

from cachepawl.integrations.vllm.advisory import JsonLike

JsonObject: TypeAlias = dict[str, object]
ReadinessClassification: TypeAlias = Literal[
    "ready_for_controlled_substitution",
    "blocked_missing_invariants",
    "advisory_only_recommended",
]
CheckStatus: TypeAlias = Literal["passed", "failed", "blocked"]


@dataclass(frozen=True, slots=True)
class VllmMutationReadinessCheck:
    """One compatibility or readiness check."""

    name: str
    status: CheckStatus
    details: str
    required_for_mutation: bool = True

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "name": self.name,
            "status": self.status,
            "details": self.details,
            "required_for_mutation": self.required_for_mutation,
        }


@dataclass(frozen=True, slots=True)
class VllmMutationReadinessReport:
    """Serializable pre-mutation readiness report."""

    classification: ReadinessClassification
    ready_for_controlled_substitution: bool
    advisory_only: bool
    non_mutating: bool
    returned_to_vllm: bool
    vllm_behavior_changed: bool
    passed_invariants: tuple[str, ...]
    failed_invariants: tuple[str, ...]
    blocked_invariants: tuple[str, ...]
    mutation_required_missing_fields: tuple[str, ...]
    checks: tuple[VllmMutationReadinessCheck, ...]

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "classification": self.classification,
            "ready_for_controlled_substitution": self.ready_for_controlled_substitution,
            "advisory_only": self.advisory_only,
            "non_mutating": self.non_mutating,
            "returned_to_vllm": self.returned_to_vllm,
            "vllm_behavior_changed": self.vllm_behavior_changed,
            "passed_invariants": self.passed_invariants,
            "failed_invariants": self.failed_invariants,
            "blocked_invariants": self.blocked_invariants,
            "mutation_required_missing_fields": self.mutation_required_missing_fields,
            "checks": tuple(check.to_dict() for check in self.checks),
        }


def check_vllm_mutation_readiness(
    translated_planner_stage_config: object,
    advisory_diff_report: object,
    *,
    group_level_diff: object | None = None,
) -> VllmMutationReadinessReport:
    """Check whether planner/advisory artifacts are ready for mutation."""

    planner = _as_mapping(translated_planner_stage_config, "translated_planner_stage_config")
    diff = _as_mapping(advisory_diff_report, "advisory_diff_report")
    group_diff = _as_sequence(group_level_diff, "group_level_diff") if group_level_diff else ()

    checks = (
        _check_schema(planner, diff),
        _check_num_blocks(planner, diff),
        _check_group_count(planner, diff, group_diff),
        _check_tensor_count(planner, diff),
        _check_layer_coverage(planner, diff),
        _check_dtype_compatibility(planner),
        _check_block_page_size(planner, group_diff),
        _check_mamba_shapes(planner),
        _check_group_mapping(planner),
        _check_byte_consistency(planner, diff, group_diff),
        _check_missing_mutation_fields(diff),
    )
    failed = tuple(check.name for check in checks if check.status == "failed")
    blocked = tuple(check.name for check in checks if check.status == "blocked")
    passed = tuple(check.name for check in checks if check.status == "passed")
    missing_fields = tuple(
        str(item)
        for item in _as_sequence(
            diff.get("missing_fields_that_prevent_mutation"),
            "missing_fields_that_prevent_mutation",
        )
    )
    if failed:
        classification: ReadinessClassification = "blocked_missing_invariants"
    elif blocked:
        classification = "advisory_only_recommended"
    else:
        classification = "ready_for_controlled_substitution"
    return VllmMutationReadinessReport(
        classification=classification,
        ready_for_controlled_substitution=classification == "ready_for_controlled_substitution",
        advisory_only=classification != "ready_for_controlled_substitution",
        non_mutating=True,
        returned_to_vllm=False,
        vllm_behavior_changed=False,
        passed_invariants=passed,
        failed_invariants=failed,
        blocked_invariants=blocked,
        mutation_required_missing_fields=missing_fields,
        checks=checks,
    )


def _check_schema(planner: JsonObject, diff: JsonObject) -> VllmMutationReadinessCheck:
    required_planner = ("num_blocks", "groups", "tensors", "group_count", "layer_count")
    required_diff = (
        "vanilla_reserved_bytes",
        "vanilla_useful_bytes",
        "cachepawl_proposed_reserved_bytes",
        "estimated_savings_bytes",
        "missing_fields_that_prevent_mutation",
    )
    missing = [
        *(f"planner.{key}" for key in required_planner if key not in planner),
        *(f"diff.{key}" for key in required_diff if key not in diff),
    ]
    return _check(
        "planner_output_schema_compatibility",
        not missing,
        "required serialized planner and diff fields are present"
        if not missing
        else f"missing fields: {', '.join(missing)}",
    )


def _check_num_blocks(planner: JsonObject, diff: JsonObject) -> VllmMutationReadinessCheck:
    planner_blocks = _optional_int(planner.get("num_blocks"), "planner.num_blocks")
    diff_blocks = _optional_int(diff.get("num_blocks"), "diff.num_blocks")
    return _check(
        "num_blocks_compatibility",
        planner_blocks is not None and planner_blocks == diff_blocks,
        f"planner num_blocks={planner_blocks}, diff num_blocks={diff_blocks}",
    )


def _check_group_count(
    planner: JsonObject,
    diff: JsonObject,
    group_diff: tuple[object, ...],
) -> VllmMutationReadinessCheck:
    groups = _as_sequence(planner.get("groups"), "planner.groups")
    planner_count = _optional_int(planner.get("group_count"), "planner.group_count")
    diff_count = _optional_int(diff.get("cache_group_count"), "diff.cache_group_count")
    group_diff_ok = not group_diff or len(group_diff) == len(groups)
    ok = planner_count == len(groups) == diff_count and group_diff_ok
    return _check(
        "cache_group_count_compatibility",
        ok,
        (
            f"planner group_count={planner_count}, groups={len(groups)}, "
            f"diff cache_group_count={diff_count}, group_level_diff={len(group_diff)}"
        ),
    )


def _check_tensor_count(planner: JsonObject, diff: JsonObject) -> VllmMutationReadinessCheck:
    tensors = _as_sequence(planner.get("tensors"), "planner.tensors")
    diff_count = _optional_int(diff.get("cache_tensor_count"), "diff.cache_tensor_count")
    return _check(
        "cache_tensor_count_compatibility",
        len(tensors) == diff_count,
        f"planner tensors={len(tensors)}, diff cache_tensor_count={diff_count}",
    )


def _check_layer_coverage(planner: JsonObject, diff: JsonObject) -> VllmMutationReadinessCheck:
    groups = _as_sequence(planner.get("groups"), "planner.groups")
    configured_layer_count = _optional_int(planner.get("layer_count"), "planner.layer_count")
    diff_layer_count = _optional_int(diff.get("layer_count"), "diff.layer_count")
    summed_layers = 0
    named_layers: set[str] = set()
    for group in groups:
        mapping = _as_mapping(group, "planner.group")
        layer_count = _optional_int(mapping.get("layer_count"), "group.layer_count") or 0
        summed_layers += layer_count
        named_layers.update(
            str(layer) for layer in _as_sequence(mapping.get("layer_names"), "layer_names")
        )
    ok = configured_layer_count == diff_layer_count == summed_layers == len(named_layers)
    return _check(
        "layer_coverage_compatibility",
        ok,
        (
            f"planner layer_count={configured_layer_count}, diff layer_count={diff_layer_count}, "
            f"summed_layers={summed_layers}, unique_named_layers={len(named_layers)}"
        ),
    )


def _check_dtype_compatibility(planner: JsonObject) -> VllmMutationReadinessCheck:
    missing: list[str] = []
    for group in _as_sequence(planner.get("groups"), "planner.groups"):
        mapping = _as_mapping(group, "planner.group")
        spec = _as_mapping(mapping.get("cache_spec"), "group.cache_spec")
        kind = str(spec.get("cache_kind"))
        if kind == "attention" and spec.get("dtype") is None:
            missing.append(f"group {mapping.get('group_index')} attention dtype")
        if kind == "mamba":
            metadata = _as_mapping(spec.get("metadata"), "mamba.metadata")
            dtypes = _as_sequence(metadata.get("dtypes"), "mamba.metadata.dtypes")
            if not dtypes or any(dtype is None for dtype in dtypes):
                missing.append(f"group {mapping.get('group_index')} mamba state dtypes")
    return _check(
        "dtype_state_dtype_compatibility",
        not missing,
        "attention dtype and Mamba state dtypes are present"
        if not missing
        else f"missing: {', '.join(missing)}",
    )


def _check_block_page_size(
    planner: JsonObject,
    group_diff: tuple[object, ...],
) -> VllmMutationReadinessCheck:
    diff_by_index: dict[int | None, JsonObject] = {}
    for item in group_diff:
        mapping = _as_mapping(item, "group_diff")
        group_index = _optional_int(mapping.get("group_index"), "group_diff.group_index")
        diff_by_index[group_index] = mapping
    missing: list[str] = []
    for group in _as_sequence(planner.get("groups"), "planner.groups"):
        mapping = _as_mapping(group, "planner.group")
        spec = _as_mapping(mapping.get("cache_spec"), "group.cache_spec")
        group_index = _optional_int(mapping.get("group_index"), "group.group_index")
        for field in ("block_size", "page_size_bytes", "useful_bytes"):
            value = _optional_int(spec.get(field), f"group.cache_spec.{field}")
            if value is None or value <= 0:
                missing.append(f"group {group_index} {field}")
            diff_item = diff_by_index.get(group_index)
            diff_value = (
                _optional_int(diff_item.get(field), f"group_diff.{field}")
                if diff_item is not None
                else value
            )
            if diff_item is not None and value != diff_value:
                missing.append(f"group {group_index} {field} mismatch")
    return _check(
        "block_page_size_compatibility",
        not missing,
        "block size, page size, and useful bytes are present and match group diff"
        if not missing
        else f"issues: {', '.join(missing)}",
    )


def _check_mamba_shapes(planner: JsonObject) -> VllmMutationReadinessCheck:
    missing: list[str] = []
    for group in _as_sequence(planner.get("groups"), "planner.groups"):
        mapping = _as_mapping(group, "planner.group")
        spec = _as_mapping(mapping.get("cache_spec"), "group.cache_spec")
        if spec.get("cache_kind") != "mamba":
            continue
        metadata = _as_mapping(spec.get("metadata"), "mamba.metadata")
        shapes = _as_sequence(metadata.get("shapes"), "mamba.metadata.shapes")
        if not shapes:
            missing.append(f"group {mapping.get('group_index')} shapes")
            continue
        for shape in shapes:
            dims = _as_sequence(shape, "mamba shape")
            if not dims or any(_optional_int(dim, "shape dim") is None for dim in dims):
                missing.append(f"group {mapping.get('group_index')} invalid shape")
    return _check(
        "mamba_state_shape_compatibility",
        not missing,
        "Mamba state shapes are present" if not missing else f"issues: {', '.join(missing)}",
    )


def _check_group_mapping(planner: JsonObject) -> VllmMutationReadinessCheck:
    groups = _as_sequence(planner.get("groups"), "planner.groups")
    attention_count = 0
    mamba_count = 0
    issues: list[str] = []
    for group in groups:
        mapping = _as_mapping(group, "planner.group")
        spec = _as_mapping(mapping.get("cache_spec"), "group.cache_spec")
        kind = spec.get("cache_kind")
        spec_type = spec.get("spec_type")
        if kind == "attention":
            attention_count += 1
            if spec_type != "FullAttentionSpec":
                issues.append(f"group {mapping.get('group_index')} attention spec_type={spec_type}")
        elif kind == "mamba":
            mamba_count += 1
            if spec_type != "MambaSpec":
                issues.append(f"group {mapping.get('group_index')} mamba spec_type={spec_type}")
        else:
            issues.append(f"group {mapping.get('group_index')} cache_kind={kind}")
    expected_attention = _optional_int(
        planner.get("attention_group_count"),
        "attention_group_count",
    )
    expected_mamba = _optional_int(planner.get("mamba_group_count"), "mamba_group_count")
    ok = not issues and attention_count == expected_attention and mamba_count == expected_mamba
    return _check(
        "attention_mamba_group_mapping_compatibility",
        ok,
        (
            f"attention_groups={attention_count}/{expected_attention}, "
            f"mamba_groups={mamba_count}/{expected_mamba}"
            if not issues
            else f"issues: {', '.join(issues)}"
        ),
    )


def _check_byte_consistency(
    planner: JsonObject,
    diff: JsonObject,
    group_diff: tuple[object, ...],
) -> VllmMutationReadinessCheck:
    num_blocks = _optional_int(planner.get("num_blocks"), "planner.num_blocks")
    vanilla_reserved = _sum_tensor_bytes(planner)
    useful_per_block = _optional_int(planner.get("total_useful_bytes"), "total_useful_bytes")
    vanilla_useful = (
        useful_per_block * num_blocks
        if useful_per_block is not None and num_blocks is not None
        else None
    )
    proposed = diff.get("cachepawl_proposed_reserved_bytes")
    savings = diff.get("estimated_savings_bytes")
    ok = False
    if (
        vanilla_reserved is not None
        and vanilla_useful is not None
        and isinstance(proposed, int)
        and isinstance(savings, int)
    ):
        ok = (
            vanilla_reserved == diff.get("vanilla_reserved_bytes")
            and vanilla_useful == diff.get("vanilla_useful_bytes")
            and proposed == vanilla_useful
            and savings == vanilla_reserved - proposed
        )
    group_ok = _group_byte_consistency(group_diff)
    return _check(
        "estimated_bytes_useful_bytes_consistency",
        ok and group_ok,
        (
            f"reserved={vanilla_reserved}, useful={vanilla_useful}, "
            f"proposed={proposed}, savings={savings}, group_level={group_ok}"
        ),
    )


def _check_missing_mutation_fields(diff: JsonObject) -> VllmMutationReadinessCheck:
    missing = tuple(
        str(item)
        for item in _as_sequence(
            diff.get("missing_fields_that_prevent_mutation"),
            "missing_fields_that_prevent_mutation",
        )
    )
    if missing:
        return VllmMutationReadinessCheck(
            name="mutation_required_missing_fields",
            status="blocked",
            details="; ".join(missing),
            required_for_mutation=True,
        )
    return VllmMutationReadinessCheck(
        name="mutation_required_missing_fields",
        status="passed",
        details="no mutation-blocking fields reported",
        required_for_mutation=True,
    )


def _sum_tensor_bytes(planner: JsonObject) -> int | None:
    total = 0
    for tensor in _as_sequence(planner.get("tensors"), "planner.tensors"):
        size = _optional_int(_as_mapping(tensor, "tensor").get("size_bytes"), "tensor.size_bytes")
        if size is None:
            return None
        total += size
    return total


def _group_byte_consistency(group_diff: tuple[object, ...]) -> bool:
    for item in group_diff:
        mapping = _as_mapping(item, "group_diff")
        vanilla = _optional_int(mapping.get("vanilla_reserved_bytes"), "vanilla_reserved_bytes")
        proposed = _optional_int(
            mapping.get("cachepawl_proposed_reserved_bytes"),
            "cachepawl_proposed_reserved_bytes",
        )
        savings = _optional_int(mapping.get("estimated_savings_bytes"), "estimated_savings_bytes")
        if vanilla is None or proposed is None or savings is None or vanilla - proposed != savings:
            return False
        if not _float_close(mapping.get("overestimation_ratio"), _ratio(vanilla, proposed)):
            return False
        if not _float_close(mapping.get("wasted_fraction"), _wasted_fraction(vanilla, proposed)):
            return False
    return True


def _check(name: str, ok: bool, details: str) -> VllmMutationReadinessCheck:
    return VllmMutationReadinessCheck(
        name=name,
        status="passed" if ok else "failed",
        details=details,
    )


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


def _float_close(value: object, expected: float | None) -> bool:
    if expected is None:
        return value is None
    return isinstance(value, (int, float)) and isclose(float(value), expected)


def _as_mapping(value: object, name: str) -> JsonObject:
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
