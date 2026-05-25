"""Observe runtime vLLM cache planning objects without mutating vLLM.

The helpers in this module accept duck-typed vanilla vLLM ``LLM`` objects and
never import ``vllm``. They only walk already-initialized runtime objects,
translate the resolved ``KVCacheConfig``, and return Cachepawl-owned records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from cachepawl.integrations.vllm.translator import (
    JsonLike,
    VllmTranslatedCacheConfig,
    VllmTranslationError,
    translate_kv_cache_config,
)

VllmRuntimeObservationStatus: TypeAlias = Literal[
    "runtime_resolved_translation",
    "unsupported",
]

RUNTIME_KV_CACHE_CONFIG_PATH = "LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config"

_RUNTIME_PATH_STEPS = (
    ("llm_engine", "LLM.llm_engine"),
    ("engine_core", "LLM.llm_engine.engine_core"),
    ("engine_core", "LLM.llm_engine.engine_core.engine_core"),
    ("scheduler", "LLM.llm_engine.engine_core.engine_core.scheduler"),
    ("kv_cache_config", RUNTIME_KV_CACHE_CONFIG_PATH),
)
_MISSING = object()


@dataclass(frozen=True, slots=True)
class VllmRuntimeCacheObservation:
    """Serializable observation of a resolved runtime vLLM cache plan."""

    status: VllmRuntimeObservationStatus
    runtime_path: str | None
    translated_cache_config: VllmTranslatedCacheConfig | None
    raw_safe_metadata: dict[str, JsonLike] = field(default_factory=dict)
    manager_path_matches_scheduler: bool | None = None
    unsupported_reason: str | None = None

    @property
    def object_access(self) -> dict[str, bool]:
        reached = self.status == "runtime_resolved_translation"
        return {
            "runtime_resolved_kv_cache_config": reached,
            "long_lived_serve": False,
            "allocator_replacement": False,
            "monkeypatching": False,
            "vllm_source_modified": False,
            "path_c_mutation": False,
        }

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "status": self.status,
            "runtime_path": self.runtime_path,
            "manager_path_matches_scheduler": self.manager_path_matches_scheduler,
            "translated_runtime_cache_config": (
                self.translated_cache_config.to_dict()
                if self.translated_cache_config is not None
                else None
            ),
            "raw_safe_metadata": dict(sorted(self.raw_safe_metadata.items())),
            "unsupported_reason": self.unsupported_reason,
            "object_access": self.object_access,
        }


def observe_vllm_runtime_cache_plan(llm: object) -> VllmRuntimeCacheObservation:
    """Locate and translate vanilla vLLM's resolved runtime ``KVCacheConfig``.

    Unsupported object shapes return a structured ``unsupported`` result instead
    of leaking ``AttributeError`` from private runtime paths.
    """

    resolved = _resolve_runtime_path(llm)
    if isinstance(resolved, VllmRuntimeCacheObservation):
        return resolved

    llm_engine, engine_core_client, engine_core, scheduler, kv_cache_config = resolved
    try:
        translated = translate_kv_cache_config(kv_cache_config)
    except VllmTranslationError as exc:
        return _unsupported(f"failed to translate runtime KVCacheConfig: {exc}")

    manager_config = _optional_path(scheduler, ("kv_cache_manager", "kv_cache_config"))
    metadata = _runtime_metadata(
        llm_engine=llm_engine,
        engine_core_client=engine_core_client,
        engine_core=engine_core,
        scheduler=scheduler,
        kv_cache_config=kv_cache_config,
    )

    return VllmRuntimeCacheObservation(
        status="runtime_resolved_translation",
        runtime_path=RUNTIME_KV_CACHE_CONFIG_PATH,
        translated_cache_config=translated,
        raw_safe_metadata=metadata,
        manager_path_matches_scheduler=(
            manager_config is kv_cache_config if manager_config is not _MISSING else None
        ),
    )


def _resolve_runtime_path(
    llm: object,
) -> tuple[object, object, object, object, object] | VllmRuntimeCacheObservation:
    current = llm
    resolved: list[object] = []
    for attr_name, path in _RUNTIME_PATH_STEPS:
        next_value = _optional_attr(current, attr_name)
        if next_value is _MISSING:
            return _unsupported(f"missing runtime attribute `{path}`")
        resolved.append(next_value)
        current = next_value
    return (
        resolved[0],
        resolved[1],
        resolved[2],
        resolved[3],
        resolved[4],
    )


def _runtime_metadata(
    *,
    llm_engine: object,
    engine_core_client: object,
    engine_core: object,
    scheduler: object,
    kv_cache_config: object,
) -> dict[str, JsonLike]:
    metadata: dict[str, JsonLike] = {
        "engine_core_client_type": type(engine_core_client).__name__,
        "engine_core_type": type(engine_core).__name__,
        "scheduler_type": type(scheduler).__name__,
        "kv_cache_config_type": type(kv_cache_config).__name__,
    }
    _add_optional_int(metadata, "num_blocks", _optional_attr(kv_cache_config, "num_blocks"))
    _add_optional_len(
        metadata,
        "kv_cache_group_count",
        _optional_attr(kv_cache_config, "kv_cache_groups"),
    )
    _add_optional_len(
        metadata,
        "kv_cache_tensor_count",
        _optional_attr(kv_cache_config, "kv_cache_tensors"),
    )
    _add_optional_int(
        metadata,
        "available_gpu_memory_for_kv_cache",
        _optional_attr(engine_core, "available_gpu_memory_for_kv_cache"),
    )
    cache_config = _optional_path(llm_engine, ("vllm_config", "cache_config"))
    if cache_config is not _MISSING:
        _add_optional_int(
            metadata,
            "cache_config_block_size",
            _optional_attr(cache_config, "block_size"),
        )
        _add_optional_int(
            metadata,
            "cache_config_num_gpu_blocks",
            _optional_attr(cache_config, "num_gpu_blocks"),
        )
    return metadata


def _unsupported(reason: str) -> VllmRuntimeCacheObservation:
    return VllmRuntimeCacheObservation(
        status="unsupported",
        runtime_path=None,
        translated_cache_config=None,
        unsupported_reason=reason,
    )


def _optional_path(obj: object, names: tuple[str, ...]) -> object:
    current = obj
    for name in names:
        current = _optional_attr(current, name)
        if current is _MISSING:
            return _MISSING
    return current


def _optional_attr(obj: object, name: str) -> object:
    try:
        return getattr(obj, name)
    except AttributeError:
        return _MISSING


def _add_optional_int(metadata: dict[str, JsonLike], name: str, value: object) -> None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        metadata[name] = value


def _add_optional_len(metadata: dict[str, JsonLike], name: str, value: object) -> None:
    if value is _MISSING or value is None:
        return
    try:
        metadata[name] = len(value)  # type: ignore[arg-type]
    except TypeError:
        return
