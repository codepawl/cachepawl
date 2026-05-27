"""Read-only live-request contract observations for vanilla vLLM objects.

This module is import-safe without vLLM. It drives one already-initialized LLM
through the public offline request path and records bounded scheduler/cache
metadata before, during, and after the request. It does not monkeypatch,
replace allocators, alter scheduler behavior, or return Cachepawl plans.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from cachepawl.integrations.vllm.runtime_contract import (
    _MISSING,
    RUNTIME_CONTRACT_BASE_PATH,
    _block_usage_metadata,
    _json_safe,
    _optional_attr,
    _resolve_base_path,
)
from cachepawl.integrations.vllm.translator import JsonLike

VllmLiveRequestContractStatus: TypeAlias = Literal[
    "live_request_contract_observation",
    "unsupported",
]
VllmLiveRequestContractFieldStatus: TypeAlias = Literal["observed", "blocked"]

_MAX_REQUEST_IDS = 8
_MAX_BLOCK_IDS = 16


@dataclass(frozen=True, slots=True)
class VllmLiveRequestContractField:
    """Status for one live-request runtime contract field."""

    name: str
    status: VllmLiveRequestContractFieldStatus
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
class VllmLiveRequestContractObservation:
    """Serializable observation of one bounded live vLLM request."""

    status: VllmLiveRequestContractStatus
    runtime_path: str | None
    request_id: str | None = None
    prompt: str | None = None
    max_new_tokens: int | None = None
    snapshots: tuple[dict[str, JsonLike], ...] = ()
    output_metadata: dict[str, JsonLike] = field(default_factory=dict)
    fields: tuple[VllmLiveRequestContractField, ...] = ()
    raw_safe_metadata: dict[str, JsonLike] = field(default_factory=dict)
    unsupported_reason: str | None = None

    @property
    def field_level_blockers(self) -> tuple[VllmLiveRequestContractField, ...]:
        return tuple(field for field in self.fields if field.status == "blocked")

    @property
    def object_access(self) -> dict[str, bool]:
        reached = self.status == "live_request_contract_observation"
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
        }

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "status": self.status,
            "runtime_path": self.runtime_path,
            "request_id": self.request_id,
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "snapshots": self.snapshots,
            "output_metadata": dict(sorted(self.output_metadata.items())),
            "fields": tuple(field.to_dict() for field in self.fields),
            "field_level_blockers": tuple(field.to_dict() for field in self.field_level_blockers),
            "raw_safe_metadata": dict(sorted(self.raw_safe_metadata.items())),
            "unsupported_reason": self.unsupported_reason,
            "object_access": self.object_access,
        }


def observe_vllm_live_request_contract(
    llm: object,
    *,
    prompt: str,
    sampling_params: object,
    max_new_tokens: int,
) -> VllmLiveRequestContractObservation:
    """Schedule one request and capture read-only block assignment metadata."""

    resolved = _resolve_base_path(llm)
    if not isinstance(resolved, tuple):
        return VllmLiveRequestContractObservation(
            status="unsupported",
            runtime_path=None,
            unsupported_reason=resolved.unsupported_reason,
        )

    llm_engine, engine_core_client, engine_core, scheduler = resolved
    manager = _optional_attr(scheduler, "kv_cache_manager")
    snapshots: list[dict[str, JsonLike]] = [
        _snapshot(llm, scheduler, manager, request_id=None, phase="before_enqueue")
    ]

    enqueue = _optional_attr(llm, "enqueue")
    if not callable(enqueue):
        return _unsupported("LLM.enqueue was not reachable")
    try:
        request_ids_obj = enqueue(
            [prompt],
            sampling_params,
            use_tqdm=False,
        )
    except Exception as exc:
        return _unsupported(f"LLM.enqueue failed: {type(exc).__name__}: {exc}")

    request_id = _first_request_id(request_ids_obj)
    if request_id is None:
        return _unsupported("LLM.enqueue did not return a request id")

    snapshots.append(
        _snapshot(llm, scheduler, manager, request_id=request_id, phase="after_enqueue_before_step")
    )

    step = _optional_path(llm, ("llm_engine", "step"))
    step_error: str | None = None
    step_output_metadata: dict[str, JsonLike] = {}
    if callable(step):
        try:
            step_outputs = step()
            step_output_metadata = _outputs_metadata(step_outputs, label="first_step")
        except Exception as exc:
            step_error = f"{type(exc).__name__}: {exc}"
    else:
        step_error = "LLM.llm_engine.step was not reachable"
    snapshots.append(
        _snapshot(llm, scheduler, manager, request_id=request_id, phase="after_first_step")
    )

    outputs_obj: object = ()
    wait_error: str | None = None
    wait_for_completion = _optional_attr(llm, "wait_for_completion")
    if callable(wait_for_completion):
        try:
            outputs_obj = wait_for_completion(use_tqdm=False)
        except Exception as exc:
            wait_error = f"{type(exc).__name__}: {exc}"
    else:
        wait_error = "LLM.wait_for_completion was not reachable"
    snapshots.append(
        _snapshot(llm, scheduler, manager, request_id=request_id, phase="after_completion")
    )

    output_metadata = {
        **step_output_metadata,
        **_outputs_metadata(outputs_obj, label="completion"),
        "first_step_error": step_error,
        "completion_error": wait_error,
    }
    fields = _live_request_fields(
        request_id=request_id,
        snapshots=tuple(snapshots),
        output_metadata=output_metadata,
    )
    return VllmLiveRequestContractObservation(
        status="live_request_contract_observation",
        runtime_path=RUNTIME_CONTRACT_BASE_PATH,
        request_id=request_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        snapshots=tuple(snapshots),
        output_metadata=output_metadata,
        fields=fields,
        raw_safe_metadata={
            "llm_engine_type": type(llm_engine).__name__,
            "engine_core_client_type": type(engine_core_client).__name__,
            "engine_core_type": type(engine_core).__name__,
            "scheduler_type": type(scheduler).__name__,
            "kv_cache_manager_type": None if manager is _MISSING else type(manager).__name__,
        },
    )


def _unsupported(reason: str) -> VllmLiveRequestContractObservation:
    return VllmLiveRequestContractObservation(
        status="unsupported",
        runtime_path=None,
        unsupported_reason=reason,
    )


def _snapshot(
    llm: object,
    scheduler: object,
    manager: object,
    *,
    request_id: str | None,
    phase: str,
) -> dict[str, JsonLike]:
    return {
        "phase": phase,
        "request_id": request_id,
        "block_usage": _block_usage_metadata(manager),
        "request_block_ids": _request_block_ids(manager, request_id),
        "scheduler_request_metadata": _scheduler_request_metadata(scheduler, request_id),
        "unfinished_request_count": _safe_call_json(
            _optional_path(llm, ("llm_engine", "get_num_unfinished_requests"))
        ),
        "has_unfinished_requests": _safe_call_json(
            _optional_path(llm, ("llm_engine", "has_unfinished_requests"))
        ),
    }


def _request_block_ids(manager: object, request_id: str | None) -> dict[str, JsonLike]:
    metadata: dict[str, JsonLike] = {
        "request_id": request_id,
        "get_block_ids_callable": False,
        "status": "not_attempted",
        "block_group_count": 0,
        "total_block_id_count": 0,
        "sample_block_ids": (),
        "error": None,
    }
    get_block_ids = (
        _optional_attr(manager, "get_block_ids") if manager is not _MISSING else _MISSING
    )
    metadata["get_block_ids_callable"] = callable(get_block_ids)
    if request_id is None:
        metadata["status"] = "missing_request_id"
        return metadata
    if not callable(get_block_ids):
        metadata["status"] = "missing_get_block_ids"
        return metadata
    try:
        block_ids_obj = get_block_ids(request_id)
    except Exception as exc:
        metadata["status"] = "error"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        return metadata
    groups = _block_id_groups(block_ids_obj)
    metadata["status"] = "observed"
    metadata["block_group_count"] = len(groups)
    metadata["block_id_counts"] = tuple(len(group) for group in groups)
    metadata["total_block_id_count"] = sum(len(group) for group in groups)
    metadata["sample_block_ids"] = tuple(block_id for group in groups for block_id in group)[
        :_MAX_BLOCK_IDS
    ]
    return metadata


def _block_id_groups(value: object) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for group in value:
            ids: list[int] = []
            if isinstance(group, Iterable) and not isinstance(group, (str, bytes)):
                for item in group:
                    if isinstance(item, int) and not isinstance(item, bool):
                        ids.append(item)
            groups.append(tuple(ids))
    return tuple(groups)


def _scheduler_request_metadata(scheduler: object, request_id: str | None) -> dict[str, JsonLike]:
    metadata: dict[str, JsonLike] = {
        "request_id": request_id,
        "scheduler_type": type(scheduler).__name__,
    }
    for attr_name in (
        "requests",
        "running",
        "waiting",
        "skipped_waiting",
        "prev_step_scheduled_req_ids",
        "finished_req_ids",
    ):
        value = _optional_attr(scheduler, attr_name)
        if value is _MISSING:
            metadata[f"{attr_name}_reachable"] = False
            continue
        metadata[f"{attr_name}_reachable"] = True
        metadata[f"{attr_name}_count"] = _safe_len(value)
        ids = _request_ids_from_collection(value)
        metadata[f"{attr_name}_sample_request_ids"] = ids[:_MAX_REQUEST_IDS]
        metadata[f"{attr_name}_contains_request"] = request_id in ids if request_id else False
    return metadata


def _request_ids_from_collection(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(str(key) for key in value)
    ids: list[str] = []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for item in value:
            item_request_id = _optional_attr(item, "request_id")
            ids.append(str(item if item_request_id is _MISSING else item_request_id))
    return tuple(ids)


def _outputs_metadata(outputs: object, *, label: str) -> dict[str, JsonLike]:
    if not isinstance(outputs, Iterable) or isinstance(outputs, (str, bytes)):
        return {
            f"{label}_output_count": 0,
            f"{label}_request_ids": (),
            f"{label}_finished_count": 0,
        }
    request_ids: list[str] = []
    finished_count = 0
    token_counts: list[int] = []
    for output in outputs:
        output_request_id = _optional_attr(output, "request_id")
        if output_request_id is not _MISSING:
            request_ids.append(str(output_request_id))
        finished = _optional_attr(output, "finished")
        if isinstance(finished, bool) and finished:
            finished_count += 1
        generated = _optional_attr(output, "outputs")
        token_counts.extend(_generated_token_counts(generated))
    return {
        f"{label}_output_count": len(request_ids),
        f"{label}_request_ids": tuple(request_ids),
        f"{label}_finished_count": finished_count,
        f"{label}_generated_token_counts": tuple(token_counts),
    }


def _generated_token_counts(outputs: object) -> tuple[int, ...]:
    counts: list[int] = []
    if isinstance(outputs, Iterable) and not isinstance(outputs, (str, bytes)):
        for output in outputs:
            token_ids = _optional_attr(output, "token_ids")
            length = _safe_len(token_ids)
            if length is not None:
                counts.append(length)
    return tuple(counts)


def _live_request_fields(
    *,
    request_id: str,
    snapshots: tuple[dict[str, JsonLike], ...],
    output_metadata: dict[str, JsonLike],
) -> tuple[VllmLiveRequestContractField, ...]:
    active_block_snapshots = tuple(
        snapshot
        for snapshot in snapshots
        if _block_total(snapshot) > 0 and snapshot.get("phase") != "after_completion"
    )
    before_usage = _usage_snapshot(snapshots, "before_enqueue")
    after_usage = _usage_snapshot(snapshots, "after_completion")
    scheduler_contains = any(_scheduler_contains_request(snapshot) for snapshot in snapshots)
    fields = [
        _field(
            "live_request_id",
            bool(request_id),
            {
                "request_id": request_id,
                "completion_request_ids": output_metadata.get("completion_request_ids", ()),
            },
            "no live request id was safely observed",
        ),
        _field(
            "request_to_block_assignment",
            bool(active_block_snapshots),
            {
                "request_id": request_id,
                "active_snapshot_phases": tuple(
                    str(snapshot.get("phase")) for snapshot in active_block_snapshots
                ),
                "active_total_block_id_counts": tuple(
                    _block_total(snapshot) for snapshot in active_block_snapshots
                ),
                "active_sample_block_ids": tuple(
                    block_id
                    for snapshot in active_block_snapshots
                    for block_id in _block_samples(snapshot)
                )[:_MAX_BLOCK_IDS],
                "post_completion_total_block_id_count": _block_total_for_phase(
                    snapshots, "after_completion"
                ),
            },
            "no non-empty block id assignment was reachable while the request was live",
        ),
        _field(
            "scheduler_request_metadata",
            scheduler_contains,
            {
                "request_id": request_id,
                "phases_with_request": tuple(
                    str(snapshot.get("phase"))
                    for snapshot in snapshots
                    if _scheduler_contains_request(snapshot)
                ),
            },
            "scheduler request collections did not expose the observed request id",
        ),
        _field(
            "before_after_block_pool_usage",
            bool(before_usage) and bool(after_usage),
            {
                "before_enqueue": before_usage,
                "after_completion": after_usage,
                "usage_changed": before_usage != after_usage,
            },
            "before/after block-pool usage snapshots were not safely reachable",
        ),
    ]
    return tuple(fields)


def _field(
    name: str,
    observed: bool,
    evidence: dict[str, JsonLike],
    blocker_reason: str,
) -> VllmLiveRequestContractField:
    return VllmLiveRequestContractField(
        name=name,
        status="observed" if observed else "blocked",
        evidence=evidence,
        blocker_reason=None if observed else blocker_reason,
    )


def _usage_snapshot(
    snapshots: tuple[dict[str, JsonLike], ...],
    phase: str,
) -> dict[str, JsonLike]:
    for snapshot in snapshots:
        if snapshot.get("phase") == phase:
            value = snapshot.get("block_usage")
            if isinstance(value, dict):
                return dict(value)
    return {}


def _block_total_for_phase(
    snapshots: tuple[dict[str, JsonLike], ...],
    phase: str,
) -> int:
    for snapshot in snapshots:
        if snapshot.get("phase") == phase:
            return _block_total(snapshot)
    return 0


def _block_total(snapshot: dict[str, JsonLike]) -> int:
    request_block_ids = snapshot.get("request_block_ids")
    if not isinstance(request_block_ids, dict):
        return 0
    total = request_block_ids.get("total_block_id_count")
    return total if isinstance(total, int) and not isinstance(total, bool) else 0


def _block_samples(snapshot: dict[str, JsonLike]) -> tuple[int, ...]:
    request_block_ids = snapshot.get("request_block_ids")
    if not isinstance(request_block_ids, dict):
        return ()
    samples = request_block_ids.get("sample_block_ids")
    if not isinstance(samples, Iterable) or isinstance(samples, (str, bytes)):
        return ()
    return tuple(item for item in samples if isinstance(item, int) and not isinstance(item, bool))


def _scheduler_contains_request(snapshot: dict[str, JsonLike]) -> bool:
    metadata = snapshot.get("scheduler_request_metadata")
    if not isinstance(metadata, dict):
        return False
    return any(
        value is True
        for key, value in metadata.items()
        if isinstance(key, str) and key.endswith("_contains_request")
    )


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


def _safe_call_json(func: object) -> JsonLike:
    if not callable(func):
        return None
    try:
        return _json_safe(func())
    except Exception:
        return None


def _safe_len(value: object) -> int | None:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None
