"""Read-only Mamba/attention view contract observations for vanilla vLLM."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from cachepawl.integrations.vllm.runtime_contract import (
    _MISSING,
    RUNTIME_CONTRACT_BASE_PATH,
    _candidate_worker_roots,
    _json_safe,
    _optional_attr,
    _resolve_base_path,
    _tensor_summary,
)
from cachepawl.integrations.vllm.translator import JsonLike

VllmMambaAttentionContractStatus: TypeAlias = Literal[
    "mamba_attention_contract_observation",
    "unsupported",
]
VllmMambaAttentionContractFieldStatus: TypeAlias = Literal["observed", "blocked"]

_MAX_TENSOR_SUMMARIES = 24
_MAX_MAPPING_ITEMS = 16
_MAX_GROUP_SUMMARIES = 16


@dataclass(frozen=True, slots=True)
class VllmMambaAttentionContractField:
    """Status for one Mamba/attention runtime contract field."""

    name: str
    status: VllmMambaAttentionContractFieldStatus
    evidence: dict[str, JsonLike] = field(default_factory=dict)
    blocker_reason: str | None = None

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "name": self.name,
            "status": self.status,
            "evidence": dict(sorted(self.evidence.items())),
            "blocker_reason": self.blocker_reason,
        }


@dataclass(frozen=True, slots=True)
class VllmMambaAttentionContractObservation:
    """Serializable observation of Mamba state-index and attention views."""

    status: VllmMambaAttentionContractStatus
    runtime_path: str | None
    request_id: str | None = None
    prompt: str | None = None
    max_new_tokens: int | None = None
    snapshots: tuple[dict[str, JsonLike], ...] = ()
    fields: tuple[VllmMambaAttentionContractField, ...] = ()
    raw_safe_metadata: dict[str, JsonLike] = field(default_factory=dict)
    unsupported_reason: str | None = None

    @property
    def field_level_blockers(self) -> tuple[VllmMambaAttentionContractField, ...]:
        return tuple(field for field in self.fields if field.status == "blocked")

    @property
    def object_access(self) -> dict[str, bool]:
        reached = self.status == "mamba_attention_contract_observation"
        return {
            "runtime_contract_objects_reached": reached,
            "live_request_scheduled": self.request_id is not None,
            "long_lived_serve": False,
            "allocator_replacement": False,
            "monkeypatching": False,
            "vllm_source_modified": False,
            "scheduler_mutation": False,
            "worker_layout_mutation": False,
            "returned_to_vllm": False,
            "controlled_substitution": False,
            "tensor_serialization": False,
        }

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "status": self.status,
            "runtime_path": self.runtime_path,
            "request_id": self.request_id,
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "snapshots": self.snapshots,
            "fields": tuple(field.to_dict() for field in self.fields),
            "field_level_blockers": tuple(field.to_dict() for field in self.field_level_blockers),
            "raw_safe_metadata": dict(sorted(self.raw_safe_metadata.items())),
            "unsupported_reason": self.unsupported_reason,
            "object_access": self.object_access,
        }


def observe_vllm_mamba_attention_contract(
    llm: object,
    *,
    prompt: str,
    sampling_params: object,
    max_new_tokens: int,
) -> VllmMambaAttentionContractObservation:
    """Schedule one request and inspect safe Mamba/attention metadata."""

    resolved = _resolve_base_path(llm)
    if not isinstance(resolved, tuple):
        return VllmMambaAttentionContractObservation(
            status="unsupported",
            runtime_path=None,
            unsupported_reason=resolved.unsupported_reason,
        )

    llm_engine, engine_core_client, engine_core, scheduler = resolved
    snapshots: list[dict[str, JsonLike]] = [
        _snapshot(llm_engine, engine_core, request_id=None, phase="before_enqueue")
    ]

    enqueue = _optional_attr(llm, "enqueue")
    if not callable(enqueue):
        return _unsupported("LLM.enqueue was not reachable")
    try:
        request_ids_obj = enqueue([prompt], sampling_params, use_tqdm=False)
    except Exception as exc:
        return _unsupported(f"LLM.enqueue failed: {type(exc).__name__}: {exc}")
    request_id = _first_request_id(request_ids_obj)
    if request_id is None:
        return _unsupported("LLM.enqueue did not return a request id")

    snapshots.append(
        _snapshot(llm_engine, engine_core, request_id=request_id, phase="after_enqueue_before_step")
    )

    step = _optional_path(llm, ("llm_engine", "step"))
    first_step_error: str | None = None
    if callable(step):
        try:
            step()
        except Exception as exc:
            first_step_error = f"{type(exc).__name__}: {exc}"
    else:
        first_step_error = "LLM.llm_engine.step was not reachable"
    snapshots.append(
        _snapshot(llm_engine, engine_core, request_id=request_id, phase="after_first_step")
    )

    wait_error: str | None = None
    wait_for_completion = _optional_attr(llm, "wait_for_completion")
    if callable(wait_for_completion):
        try:
            wait_for_completion(use_tqdm=False)
        except Exception as exc:
            wait_error = f"{type(exc).__name__}: {exc}"
    else:
        wait_error = "LLM.wait_for_completion was not reachable"
    snapshots.append(
        _snapshot(llm_engine, engine_core, request_id=request_id, phase="after_completion")
    )

    fields = _contract_fields(request_id=request_id, snapshots=tuple(snapshots))
    return VllmMambaAttentionContractObservation(
        status="mamba_attention_contract_observation",
        runtime_path=RUNTIME_CONTRACT_BASE_PATH,
        request_id=request_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        snapshots=tuple(snapshots),
        fields=fields,
        raw_safe_metadata={
            "llm_engine_type": type(llm_engine).__name__,
            "engine_core_client_type": type(engine_core_client).__name__,
            "engine_core_type": type(engine_core).__name__,
            "scheduler_type": type(scheduler).__name__,
            "first_step_error": first_step_error,
            "completion_error": wait_error,
        },
    )


def _unsupported(reason: str) -> VllmMambaAttentionContractObservation:
    return VllmMambaAttentionContractObservation(
        status="unsupported",
        runtime_path=None,
        unsupported_reason=reason,
    )


def _snapshot(
    llm_engine: object,
    engine_core: object,
    *,
    request_id: str | None,
    phase: str,
) -> dict[str, JsonLike]:
    root_summaries = tuple(
        _root_summary(root_name, root, request_id=request_id)
        for root_name, root in _candidate_worker_roots(llm_engine, engine_core)
        if root is not _MISSING
    )
    return {
        "phase": phase,
        "request_id": request_id,
        "root_summaries": root_summaries,
        "totals": _snapshot_totals(root_summaries, request_id=request_id),
    }


def _root_summary(root_name: str, root: object, *, request_id: str | None) -> dict[str, JsonLike]:
    block_tables = _optional_attr(root, "block_tables")
    input_batch = _optional_attr(root, "input_batch")
    input_batch_block_table = _optional_path(input_batch, ("block_table",))
    if block_tables is _MISSING:
        block_tables = input_batch_block_table
    attn_groups = _optional_attr(root, "attn_groups")
    mamba_state_idx = _optional_attr(root, "mamba_state_idx")
    return {
        "root_path": root_name,
        "root_type": type(root).__name__,
        "cache_config": _cache_config_summary(_optional_attr(root, "cache_config")),
        "mamba_state_idx": _mapping_summary(mamba_state_idx, request_id=request_id),
        "input_batch": _input_batch_summary(input_batch, request_id=request_id),
        "block_tables": _block_tables_summary(block_tables),
        "attention_groups": _attention_groups_summary(attn_groups),
        "mamba_related_tensors": tuple(
            _named_tensor_summaries(
                root,
                root_name,
                names=(
                    "mamba_cache",
                    "mamba_caches",
                    "mamba_states",
                    "state_cache",
                    "state_caches",
                    "state_indices_tensor_d",
                    "state_indices_tensor_p",
                ),
            )
        ),
    }


def _cache_config_summary(cache_config: object) -> dict[str, JsonLike]:
    if cache_config is _MISSING:
        return {"reachable": False}
    return {
        "reachable": True,
        "type": type(cache_config).__name__,
        "safe_attrs": _safe_scalar_attrs(
            cache_config,
            (
                "block_size",
                "mamba_block_size",
                "mamba_cache_mode",
                "mamba_cache_dtype",
                "mamba_ssm_cache_dtype",
            ),
        ),
    }


def _input_batch_summary(input_batch: object, *, request_id: str | None) -> dict[str, JsonLike]:
    if input_batch is _MISSING:
        return {"reachable": False}
    return {
        "reachable": True,
        "type": type(input_batch).__name__,
        "num_reqs": _json_safe(_optional_attr(input_batch, "num_reqs")),
        "req_ids": tuple(
            str(item) for item in _iter_limited(_optional_attr(input_batch, "req_ids"), 8)
        ),
        "req_id_to_index": _mapping_summary(
            _optional_attr(input_batch, "req_id_to_index"),
            request_id=request_id,
        ),
    }


def _block_tables_summary(block_tables: object) -> dict[str, JsonLike]:
    if block_tables is _MISSING:
        return {"reachable": False}
    summary: dict[str, JsonLike] = {
        "reachable": True,
        "type": type(block_tables).__name__,
    }
    for attr_name in (
        "block_tables",
        "input_block_tables",
        "block_table",
        "block_table_ptrs",
        "block_table_strides",
        "input_block_table_ptrs",
        "slot_mappings",
        "slot_mapping",
    ):
        value = _optional_attr(block_tables, attr_name)
        if value is _MISSING:
            summary[f"{attr_name}_reachable"] = False
            continue
        summary[f"{attr_name}_reachable"] = True
        summary[f"{attr_name}_count"] = _safe_len(value)
        summary[f"{attr_name}_tensors"] = tuple(
            _summarize_tensors(value, f"block_tables.{attr_name}")
        )
    return summary


def _attention_groups_summary(attn_groups: object) -> dict[str, JsonLike]:
    if attn_groups is _MISSING:
        return {"reachable": False}
    summary: dict[str, JsonLike] = {
        "reachable": True,
        "outer_count": _safe_len(attn_groups),
        "groups": (),
    }
    groups: list[dict[str, JsonLike]] = []
    if isinstance(attn_groups, Iterable) and not isinstance(attn_groups, (str, bytes)):
        for outer_index, group_items in enumerate(attn_groups):
            if len(groups) >= _MAX_GROUP_SUMMARIES:
                break
            if not isinstance(group_items, Iterable) or isinstance(group_items, (str, bytes)):
                continue
            for inner_index, group in enumerate(group_items):
                if len(groups) >= _MAX_GROUP_SUMMARIES:
                    break
                groups.append(_attention_group_summary(group, outer_index, inner_index))
    summary["groups"] = tuple(groups)
    return summary


def _attention_group_summary(
    group: object,
    outer_index: int,
    inner_index: int,
) -> dict[str, JsonLike]:
    metadata_builders = _optional_attr(group, "metadata_builders")
    builder_summaries: tuple[dict[str, JsonLike], ...] = ()
    if metadata_builders is not _MISSING:
        builder_summaries = tuple(
            {
                "index": index,
                "type": type(builder).__name__,
                "safe_attrs": _safe_scalar_attrs(
                    builder,
                    ("block_size", "num_heads", "head_size", "sliding_window"),
                ),
            }
            for index, builder in enumerate(_iter_limited(metadata_builders, _MAX_GROUP_SUMMARIES))
        )
    return {
        "outer_index": outer_index,
        "inner_index": inner_index,
        "group_type": type(group).__name__,
        "metadata_builders_reachable": metadata_builders is not _MISSING,
        "metadata_builder_count": _safe_len(metadata_builders),
        "metadata_builders": builder_summaries,
        "safe_attrs": _safe_scalar_attrs(group, ("prefix", "kv_cache_group_id", "layer_names")),
    }


def _mapping_summary(mapping: object, *, request_id: str | None) -> dict[str, JsonLike]:
    if mapping is _MISSING:
        return {"reachable": False, "contains_request": False}
    if not isinstance(mapping, Mapping):
        return {
            "reachable": True,
            "type": type(mapping).__name__,
            "is_mapping": False,
            "contains_request": False,
        }
    items = tuple((str(key), _json_safe(value)) for key, value in mapping.items())[
        :_MAX_MAPPING_ITEMS
    ]
    return {
        "reachable": True,
        "type": type(mapping).__name__,
        "is_mapping": True,
        "count": len(mapping),
        "contains_request": request_id in mapping if request_id is not None else False,
        "request_value": _json_safe(mapping.get(request_id)) if request_id is not None else None,
        "sample_items": items,
    }


def _named_tensor_summaries(
    root: object,
    root_name: str,
    *,
    names: tuple[str, ...],
) -> Iterable[dict[str, JsonLike]]:
    seen: set[int] = set()
    for name in names:
        value = _optional_attr(root, name)
        if value is _MISSING:
            continue
        yield from _summarize_tensors(value, f"{root_name}.{name}", seen)


def _summarize_tensors(
    value: object,
    path: str,
    seen: set[int] | None = None,
) -> Iterable[dict[str, JsonLike]]:
    seen = set() if seen is None else seen
    if len(seen) >= _MAX_TENSOR_SUMMARIES:
        return
    if _is_tensor_like(value):
        object_id = id(value)
        if object_id in seen:
            return
        seen.add(object_id)
        yield _tensor_summary(value, path)
        return
    if _has_buffer_views(value):
        for attr_name in ("gpu", "cpu", "np"):
            tensor = _optional_attr(value, attr_name)
            if tensor is not _MISSING:
                yield from _summarize_tensors(tensor, f"{path}.{attr_name}", seen)
        return
    nested_block_table = _optional_attr(value, "block_table")
    if nested_block_table is not _MISSING:
        yield from _summarize_tensors(nested_block_table, f"{path}.block_table", seen)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if len(seen) >= _MAX_TENSOR_SUMMARIES:
                return
            yield from _summarize_tensors(item, f"{path}.{key}", seen)
        return
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            if len(seen) >= _MAX_TENSOR_SUMMARIES:
                return
            yield from _summarize_tensors(item, f"{path}[{index}]", seen)


def _snapshot_totals(
    root_summaries: tuple[dict[str, JsonLike], ...],
    *,
    request_id: str | None,
) -> dict[str, JsonLike]:
    state_idx_count = 0
    state_idx_contains_request = False
    block_table_tensor_count = 0
    attention_group_count = 0
    mamba_tensor_count = 0
    for root in root_summaries:
        state_idx = root.get("mamba_state_idx")
        if isinstance(state_idx, dict) and state_idx.get("reachable") is True:
            count = state_idx.get("count")
            if isinstance(count, int):
                state_idx_count += count
            state_idx_contains_request |= state_idx.get("contains_request") is True
        block_tables = root.get("block_tables")
        if isinstance(block_tables, dict):
            block_table_tensor_count += _nested_tensor_count(block_tables)
        groups = root.get("attention_groups")
        if isinstance(groups, dict):
            group_count = groups.get("outer_count")
            if isinstance(group_count, int):
                attention_group_count += group_count
        tensors = root.get("mamba_related_tensors")
        if isinstance(tensors, tuple):
            mamba_tensor_count += len(tensors)
    return {
        "request_id": request_id,
        "mamba_state_idx_entry_count": state_idx_count,
        "mamba_state_idx_contains_request": state_idx_contains_request,
        "block_table_tensor_count": block_table_tensor_count,
        "attention_group_count": attention_group_count,
        "mamba_related_tensor_count": mamba_tensor_count,
    }


def _contract_fields(
    *,
    request_id: str,
    snapshots: tuple[dict[str, JsonLike], ...],
) -> tuple[VllmMambaAttentionContractField, ...]:
    state_idx_phases = tuple(
        str(snapshot.get("phase"))
        for snapshot in snapshots
        if _total_bool(snapshot, "mamba_state_idx_contains_request")
    )
    block_table_phases = tuple(
        str(snapshot.get("phase"))
        for snapshot in snapshots
        if _total_int(snapshot, "block_table_tensor_count") > 0
    )
    attention_group_phases = tuple(
        str(snapshot.get("phase"))
        for snapshot in snapshots
        if _total_int(snapshot, "attention_group_count") > 0
    )
    mamba_tensor_phases = tuple(
        str(snapshot.get("phase"))
        for snapshot in snapshots
        if _total_int(snapshot, "mamba_related_tensor_count") > 0
    )
    fields = [
        _field(
            "mamba_state_index_contract",
            bool(state_idx_phases),
            {
                "request_id": request_id,
                "phases_with_request_state_index": state_idx_phases,
                "max_state_index_entry_count": max(
                    (_total_int(snapshot, "mamba_state_idx_entry_count") for snapshot in snapshots),
                    default=0,
                ),
            },
            "mamba_state_idx was not reachable with the live request id",
        ),
        _field(
            "attention_block_table_view_contract",
            bool(block_table_phases),
            {
                "phases_with_block_table_tensors": block_table_phases,
                "max_block_table_tensor_count": max(
                    (_total_int(snapshot, "block_table_tensor_count") for snapshot in snapshots),
                    default=0,
                ),
            },
            "attention block-table tensors were not safely reachable",
        ),
        _field(
            "attention_metadata_builder_contract",
            bool(attention_group_phases),
            {
                "phases_with_attention_groups": attention_group_phases,
                "max_attention_group_count": max(
                    (_total_int(snapshot, "attention_group_count") for snapshot in snapshots),
                    default=0,
                ),
            },
            "attention metadata builder groups were not safely reachable",
        ),
        _field(
            "mamba_state_tensor_contract",
            bool(mamba_tensor_phases),
            {
                "phases_with_mamba_related_tensors": mamba_tensor_phases,
                "max_mamba_related_tensor_count": max(
                    (_total_int(snapshot, "mamba_related_tensor_count") for snapshot in snapshots),
                    default=0,
                ),
            },
            "Mamba state tensors were not safely reachable by stable runtime attributes",
        ),
    ]
    return tuple(fields)


def _field(
    name: str,
    observed: bool,
    evidence: dict[str, JsonLike],
    blocker_reason: str,
) -> VllmMambaAttentionContractField:
    return VllmMambaAttentionContractField(
        name=name,
        status="observed" if observed else "blocked",
        evidence=evidence,
        blocker_reason=None if observed else blocker_reason,
    )


def _total_int(snapshot: dict[str, JsonLike], key: str) -> int:
    totals = snapshot.get("totals")
    if not isinstance(totals, dict):
        return 0
    value = totals.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _total_bool(snapshot: dict[str, JsonLike], key: str) -> bool:
    totals = snapshot.get("totals")
    return isinstance(totals, dict) and totals.get(key) is True


def _nested_tensor_count(value: object) -> int:
    if isinstance(value, dict) and "shape" in value and "dtype" in value:
        return 1
    count = 0
    if isinstance(value, Mapping):
        for item in value.values():
            count += _nested_tensor_count(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            count += _nested_tensor_count(item)
    return count


def _safe_scalar_attrs(obj: object, names: tuple[str, ...]) -> dict[str, JsonLike]:
    attrs: dict[str, JsonLike] = {}
    for name in names:
        value = _optional_attr(obj, name)
        if value is _MISSING:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            attrs[name] = value
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            attrs[name] = tuple(str(item) for item in _iter_limited(value, _MAX_MAPPING_ITEMS))
        else:
            attrs[name] = str(value)
    return attrs


def _iter_limited(value: object, limit: int) -> Iterable[object]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            if index >= limit:
                return
            yield item


def _first_request_id(value: object) -> str | None:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for item in value:
            return str(item)
    return None


def _optional_path(obj: object, names: tuple[str, ...]) -> object:
    current = obj
    for name in names:
        current = _optional_attr(current, name)
        if current is _MISSING:
            return _MISSING
    return current


def _safe_len(value: object) -> int | None:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _is_tensor_like(value: object) -> bool:
    return (
        _optional_attr(value, "shape") is not _MISSING
        and _optional_attr(value, "dtype") is not _MISSING
    )


def _has_buffer_views(value: object) -> bool:
    return any(_optional_attr(value, name) is not _MISSING for name in ("gpu", "cpu", "np"))
