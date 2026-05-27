"""Read-only runtime contract observations for vanilla vLLM objects.

This module is import-safe without vLLM. It walks already-initialized,
duck-typed runtime objects and records scalar/list/tensor metadata that is safe
to serialize. It does not call mutating manager APIs, replace objects, or
return Cachepawl plans to vLLM.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from cachepawl.integrations.vllm.translator import JsonLike

VllmRuntimeContractStatus: TypeAlias = Literal[
    "runtime_contract_observation",
    "unsupported",
]
VllmRuntimeContractFieldStatus: TypeAlias = Literal["observed", "blocked"]

RUNTIME_CONTRACT_BASE_PATH = "LLM.llm_engine.engine_core.engine_core"
_MISSING = object()
_MAX_TENSOR_SUMMARIES = 32


@dataclass(frozen=True, slots=True)
class VllmRuntimeContractField:
    """Status for one mutation-readiness runtime contract field."""

    name: str
    status: VllmRuntimeContractFieldStatus
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
class VllmRuntimeContractObservation:
    """Serializable read-only observation of vLLM runtime contracts."""

    status: VllmRuntimeContractStatus
    runtime_path: str | None
    scheduler_manager: dict[str, JsonLike] = field(default_factory=dict)
    block_usage: dict[str, JsonLike] = field(default_factory=dict)
    worker_tensors: tuple[dict[str, JsonLike], ...] = ()
    mamba_attention: dict[str, JsonLike] = field(default_factory=dict)
    fields: tuple[VllmRuntimeContractField, ...] = ()
    raw_safe_metadata: dict[str, JsonLike] = field(default_factory=dict)
    unsupported_reason: str | None = None

    @property
    def field_level_blockers(self) -> tuple[VllmRuntimeContractField, ...]:
        return tuple(field for field in self.fields if field.status == "blocked")

    @property
    def object_access(self) -> dict[str, bool]:
        reached = self.status == "runtime_contract_observation"
        return {
            "runtime_contract_objects_reached": reached,
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
            "scheduler_manager": dict(sorted(self.scheduler_manager.items())),
            "block_usage": dict(sorted(self.block_usage.items())),
            "worker_tensors": self.worker_tensors,
            "mamba_attention": dict(sorted(self.mamba_attention.items())),
            "fields": tuple(field.to_dict() for field in self.fields),
            "field_level_blockers": tuple(field.to_dict() for field in self.field_level_blockers),
            "raw_safe_metadata": dict(sorted(self.raw_safe_metadata.items())),
            "unsupported_reason": self.unsupported_reason,
            "object_access": self.object_access,
        }


def observe_vllm_runtime_contracts(llm: object) -> VllmRuntimeContractObservation:
    """Capture read-only runtime contract metadata from a vanilla vLLM ``LLM``."""

    resolved = _resolve_base_path(llm)
    if isinstance(resolved, VllmRuntimeContractObservation):
        return resolved
    llm_engine, engine_core_client, engine_core, scheduler = resolved
    manager = _optional_attr(scheduler, "kv_cache_manager")
    scheduler_config = _optional_attr(scheduler, "kv_cache_config")
    manager_config = _optional_path(manager, ("kv_cache_config",))

    scheduler_manager = _scheduler_manager_metadata(
        scheduler=scheduler,
        manager=manager,
        scheduler_config=scheduler_config,
        manager_config=manager_config,
    )
    block_usage = _block_usage_metadata(manager)
    worker_tensors = tuple(_worker_tensor_summaries(llm_engine, engine_core))
    mamba_attention = _mamba_attention_metadata(llm_engine, engine_core)
    fields = _contract_fields(
        scheduler_manager=scheduler_manager,
        block_usage=block_usage,
        worker_tensors=worker_tensors,
        mamba_attention=mamba_attention,
        manager=manager,
    )

    return VllmRuntimeContractObservation(
        status="runtime_contract_observation",
        runtime_path=RUNTIME_CONTRACT_BASE_PATH,
        scheduler_manager=scheduler_manager,
        block_usage=block_usage,
        worker_tensors=worker_tensors,
        mamba_attention=mamba_attention,
        fields=fields,
        raw_safe_metadata={
            "llm_engine_type": type(llm_engine).__name__,
            "engine_core_client_type": type(engine_core_client).__name__,
            "engine_core_type": type(engine_core).__name__,
            "scheduler_type": type(scheduler).__name__,
        },
    )


def _resolve_base_path(
    llm: object,
) -> tuple[object, object, object, object] | VllmRuntimeContractObservation:
    llm_engine = _optional_attr(llm, "llm_engine")
    if llm_engine is _MISSING:
        return _unsupported("missing runtime attribute `LLM.llm_engine`")
    engine_core_client = _optional_attr(llm_engine, "engine_core")
    if engine_core_client is _MISSING:
        return _unsupported("missing runtime attribute `LLM.llm_engine.engine_core`")
    engine_core = _optional_attr(engine_core_client, "engine_core")
    if engine_core is _MISSING:
        return _unsupported("missing runtime attribute `LLM.llm_engine.engine_core.engine_core`")
    scheduler = _optional_attr(engine_core, "scheduler")
    if scheduler is _MISSING:
        return _unsupported(f"missing runtime attribute `{RUNTIME_CONTRACT_BASE_PATH}.scheduler`")
    return llm_engine, engine_core_client, engine_core, scheduler


def _unsupported(reason: str) -> VllmRuntimeContractObservation:
    return VllmRuntimeContractObservation(
        status="unsupported",
        runtime_path=None,
        unsupported_reason=reason,
    )


def _scheduler_manager_metadata(
    *,
    scheduler: object,
    manager: object,
    scheduler_config: object,
    manager_config: object,
) -> dict[str, JsonLike]:
    metadata: dict[str, JsonLike] = {
        "scheduler_type": type(scheduler).__name__,
        "scheduler_has_kv_cache_manager": manager is not _MISSING,
        "scheduler_has_kv_cache_config": scheduler_config is not _MISSING,
    }
    if manager is _MISSING:
        return metadata
    metadata.update(
        {
            "kv_cache_manager_type": type(manager).__name__,
            "manager_config_matches_scheduler": (
                manager_config is scheduler_config
                if manager_config is not _MISSING and scheduler_config is not _MISSING
                else None
            ),
        }
    )
    coordinator = _optional_attr(manager, "coordinator")
    if coordinator is not _MISSING:
        metadata["coordinator_type"] = type(coordinator).__name__
    block_pool = _optional_attr(manager, "block_pool")
    if block_pool is not _MISSING:
        metadata["block_pool_type"] = type(block_pool).__name__
    methods = tuple(
        name
        for name in (
            "allocate_slots",
            "free",
            "get_block_ids",
            "get_usage",
            "take_events",
            "take_new_block_ids",
        )
        if callable(_optional_attr(manager, name))
    )
    metadata["kv_cache_manager_methods_observed"] = methods
    return metadata


def _block_usage_metadata(manager: object) -> dict[str, JsonLike]:
    if manager is _MISSING:
        return {}
    metadata: dict[str, JsonLike] = {}
    usage = _safe_call_no_args(_optional_attr(manager, "get_usage"))
    if usage is not _MISSING:
        metadata["manager_get_usage"] = _json_safe(usage)
    block_pool = _optional_attr(manager, "block_pool")
    if block_pool is _MISSING:
        return metadata
    free_blocks = _safe_call_no_args(_optional_attr(block_pool, "get_num_free_blocks"))
    if free_blocks is not _MISSING:
        metadata["block_pool_free_blocks"] = _json_safe(free_blocks)
    for name in ("num_gpu_blocks", "num_cpu_blocks", "num_blocks"):
        value = _optional_attr(block_pool, name)
        if value is not _MISSING:
            metadata[f"block_pool_{name}"] = _json_safe(value)
    return metadata


def _worker_tensor_summaries(
    llm_engine: object, engine_core: object
) -> Iterable[dict[str, JsonLike]]:
    seen: set[int] = set()
    for root_name, root in _candidate_worker_roots(llm_engine, engine_core):
        if root is _MISSING:
            continue
        root_metadata = {"root_path": root_name, "root_type": type(root).__name__}
        for attr_name in (
            "kv_caches",
            "kv_cache",
            "kv_cache_tensors",
            "gpu_kv_caches",
            "cpu_kv_caches",
            "caches",
        ):
            value = _optional_attr(root, attr_name)
            if value is _MISSING:
                continue
            for summary in _summarize_tensors(value, f"{root_name}.{attr_name}", seen):
                yield {**root_metadata, **summary}


def _candidate_worker_roots(
    llm_engine: object, engine_core: object
) -> tuple[tuple[str, object], ...]:
    model_executor = _optional_attr(engine_core, "model_executor")
    return (
        ("LLM.llm_engine", llm_engine),
        (f"{RUNTIME_CONTRACT_BASE_PATH}.model_executor", model_executor),
        (
            f"{RUNTIME_CONTRACT_BASE_PATH}.model_executor.driver_worker",
            _optional_path(model_executor, ("driver_worker",)),
        ),
        (
            f"{RUNTIME_CONTRACT_BASE_PATH}.model_executor.driver_worker.model_runner",
            _optional_path(model_executor, ("driver_worker", "model_runner")),
        ),
        (
            f"{RUNTIME_CONTRACT_BASE_PATH}.model_executor.model_runner",
            _optional_path(model_executor, ("model_runner",)),
        ),
        (f"{RUNTIME_CONTRACT_BASE_PATH}.model_runner", _optional_attr(engine_core, "model_runner")),
    )


def _mamba_attention_metadata(llm_engine: object, engine_core: object) -> dict[str, JsonLike]:
    metadata: dict[str, JsonLike] = {}
    roots = _candidate_worker_roots(llm_engine, engine_core)
    builder_count = 0
    state_tensor_count = 0
    block_table_count = 0
    for root_name, root in roots:
        if root is _MISSING:
            continue
        for attr_name in (
            "attn_metadata_builders",
            "metadata_builders",
            "mamba_metadata_builder",
            "attn_groups",
        ):
            value = _optional_attr(root, attr_name)
            if value is not _MISSING:
                builder_count += _safe_len(value) or 1
                metadata.setdefault("metadata_builder_paths", ())
                metadata["metadata_builder_paths"] = (
                    *tuple(metadata["metadata_builder_paths"]),  # type: ignore[arg-type]
                    f"{root_name}.{attr_name}",
                )
        for attr_name in ("state_indices_tensor_d", "state_indices_tensor_p"):
            tensor = _optional_attr(root, attr_name)
            if tensor is not _MISSING and _is_tensor_like(tensor):
                state_tensor_count += 1
        for attr_name in ("block_table", "block_table_tensor", "block_tables"):
            table = _optional_attr(root, attr_name)
            if table is not _MISSING:
                block_table_count += 1
    metadata["metadata_builder_count"] = builder_count
    metadata["state_index_tensor_count"] = state_tensor_count
    metadata["block_table_attr_count"] = block_table_count
    return metadata


def _contract_fields(
    *,
    scheduler_manager: dict[str, JsonLike],
    block_usage: dict[str, JsonLike],
    worker_tensors: tuple[dict[str, JsonLike], ...],
    mamba_attention: dict[str, JsonLike],
    manager: object,
) -> tuple[VllmRuntimeContractField, ...]:
    fields: list[VllmRuntimeContractField] = []
    fields.append(
        _field(
            "scheduler_kv_cache_manager_structure",
            bool(scheduler_manager.get("scheduler_has_kv_cache_manager")),
            scheduler_manager,
            "scheduler.kv_cache_manager was not reachable",
        )
    )
    fields.append(
        _field(
            "block_usage_metadata",
            bool(block_usage),
            block_usage,
            "no manager/block-pool usage metadata was safely reachable",
        )
    )
    fields.append(
        _field(
            "worker_cache_tensor_layout",
            bool(worker_tensors),
            {"tensor_summary_count": len(worker_tensors)},
            "worker cache tensors were not reachable without private runtime traversal",
        )
    )
    request_block_observed = False
    get_block_ids = (
        _optional_attr(manager, "get_block_ids") if manager is not _MISSING else _MISSING
    )
    request_reason = "KVCacheManager.get_block_ids was not reachable"
    if callable(get_block_ids):
        request_reason = (
            "get_block_ids exists but needs a live request id; no request was "
            "scheduled in this read-only capture"
        )
    fields.append(
        _field(
            "request_to_block_assignment",
            request_block_observed,
            {"get_block_ids_callable": callable(get_block_ids)},
            request_reason,
        )
    )
    mamba_observed = bool(mamba_attention.get("state_index_tensor_count")) or bool(
        mamba_attention.get("block_table_attr_count")
    )
    fields.append(
        _field(
            "mamba_state_index_attention_view_contract",
            mamba_observed,
            mamba_attention,
            "Mamba state-index and attention block-table tensors were not safely reachable",
        )
    )
    return tuple(fields)


def _field(
    name: str,
    observed: bool,
    evidence: dict[str, JsonLike],
    blocker_reason: str,
) -> VllmRuntimeContractField:
    return VllmRuntimeContractField(
        name=name,
        status="observed" if observed else "blocked",
        evidence=evidence,
        blocker_reason=None if observed else blocker_reason,
    )


def _summarize_tensors(value: object, path: str, seen: set[int]) -> Iterable[dict[str, JsonLike]]:
    if len(seen) >= _MAX_TENSOR_SUMMARIES:
        return
    if _is_tensor_like(value):
        object_id = id(value)
        if object_id in seen:
            return
        seen.add(object_id)
        yield _tensor_summary(value, path)
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


def _tensor_summary(tensor: object, path: str) -> dict[str, JsonLike]:
    return {
        "path": path,
        "tensor_type": type(tensor).__name__,
        "shape": _shape_tuple(tensor),
        "stride": _stride_tuple(tensor),
        "dtype": _optional_string(_optional_attr(tensor, "dtype")),
        "device": _optional_string(_optional_attr(tensor, "device")),
        "layout": _optional_string(_optional_attr(tensor, "layout")),
    }


def _is_tensor_like(value: object) -> bool:
    return (
        _optional_attr(value, "shape") is not _MISSING
        and _optional_attr(value, "dtype") is not _MISSING
    )


def _shape_tuple(value: object) -> tuple[int, ...] | None:
    shape = _optional_attr(value, "shape")
    if shape is _MISSING:
        return None
    return _int_tuple(shape)


def _stride_tuple(value: object) -> tuple[int, ...] | None:
    stride = _optional_attr(value, "stride")
    if callable(stride):
        stride = _safe_call_no_args(stride)
    if stride is _MISSING:
        return None
    return _int_tuple(stride)


def _int_tuple(value: object) -> tuple[int, ...] | None:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        items: list[int] = []
        for item in value:
            if isinstance(item, int) and not isinstance(item, bool):
                items.append(item)
            else:
                return None
        return tuple(items)
    return None


def _optional_path(obj: object, names: tuple[str, ...]) -> object:
    current = obj
    if current is _MISSING:
        return _MISSING
    for name in names:
        current = _optional_attr(current, name)
        if current is _MISSING:
            return _MISSING
    return current


def _optional_attr(obj: object, name: str) -> object:
    if obj is _MISSING:
        return _MISSING
    try:
        return getattr(obj, name)
    except AttributeError:
        return _MISSING


def _safe_call_no_args(func: object) -> object:
    if not callable(func):
        return _MISSING
    try:
        return func()
    except Exception:
        return _MISSING


def _safe_len(value: object) -> int | None:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _optional_string(value: object) -> str | None:
    if value is _MISSING or value is None:
        return None
    return str(value)


def _json_safe(value: object) -> JsonLike:
    if value is _MISSING:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return tuple(_json_safe(item) for item in value)
    return str(value)
