"""Tests for observe-first vLLM runtime cache-plan observation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from cachepawl.integrations.vllm import (
    RUNTIME_KV_CACHE_CONFIG_PATH,
    observe_vllm_runtime_cache_plan,
)


@dataclass(frozen=True, slots=True)
class FakeAttentionSpec:
    block_size: int
    page_size_bytes: int
    real_page_size_bytes: int
    num_kv_heads: int
    head_size: int
    dtype: str


@dataclass(frozen=True, slots=True)
class FakeKVCacheGroupSpec:
    layer_names: tuple[str, ...]
    kv_cache_spec: object
    group_id: str | None = None


@dataclass(frozen=True, slots=True)
class FakeKVCacheTensor:
    size: int
    shared_by: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FakeKVCacheConfig:
    num_blocks: int
    kv_cache_groups: tuple[FakeKVCacheGroupSpec, ...]
    kv_cache_tensors: tuple[FakeKVCacheTensor, ...]


@dataclass(frozen=True, slots=True)
class FakeCacheConfig:
    block_size: int
    num_gpu_blocks: int


@dataclass(frozen=True, slots=True)
class FakeVllmConfig:
    cache_config: FakeCacheConfig


@dataclass(frozen=True, slots=True)
class FakeKVCacheManager:
    kv_cache_config: FakeKVCacheConfig


@dataclass(frozen=True, slots=True)
class FakeScheduler:
    kv_cache_config: FakeKVCacheConfig
    kv_cache_manager: FakeKVCacheManager


@dataclass(frozen=True, slots=True)
class FakeEngineCore:
    scheduler: FakeScheduler
    available_gpu_memory_for_kv_cache: int


@dataclass(frozen=True, slots=True)
class FakeEngineCoreClient:
    engine_core: FakeEngineCore


@dataclass(frozen=True, slots=True)
class FakeLLMEngine:
    engine_core: FakeEngineCoreClient
    vllm_config: FakeVllmConfig


@dataclass(frozen=True, slots=True)
class FakeLLM:
    llm_engine: FakeLLMEngine


def test_observer_translates_known_runtime_path_without_vllm_dependency() -> None:
    config = FakeKVCacheConfig(
        num_blocks=329,
        kv_cache_groups=(
            FakeKVCacheGroupSpec(
                layer_names=("model.attn.0", "model.attn.1"),
                kv_cache_spec=FakeAttentionSpec(
                    block_size=48,
                    page_size_bytes=983_040,
                    real_page_size_bytes=983_040,
                    num_kv_heads=32,
                    head_size=160,
                    dtype="torch.bfloat16",
                ),
                group_id="attention",
            ),
        ),
        kv_cache_tensors=(FakeKVCacheTensor(size=323_420_160, shared_by=("model.attn.0",)),),
    )
    llm = FakeLLM(
        llm_engine=FakeLLMEngine(
            engine_core=FakeEngineCoreClient(
                engine_core=FakeEngineCore(
                    scheduler=FakeScheduler(
                        kv_cache_config=config,
                        kv_cache_manager=FakeKVCacheManager(kv_cache_config=config),
                    ),
                    available_gpu_memory_for_kv_cache=2_915_421_184,
                )
            ),
            vllm_config=FakeVllmConfig(
                cache_config=FakeCacheConfig(block_size=48, num_gpu_blocks=329)
            ),
        )
    )

    observed = observe_vllm_runtime_cache_plan(llm)
    payload = observed.to_dict()
    object_access = cast(dict[str, object], payload["object_access"])

    assert observed.status == "runtime_resolved_translation"
    assert observed.runtime_path == RUNTIME_KV_CACHE_CONFIG_PATH
    assert observed.manager_path_matches_scheduler is True
    assert object_access["runtime_resolved_kv_cache_config"] is True
    assert payload["raw_safe_metadata"] == {
        "available_gpu_memory_for_kv_cache": 2_915_421_184,
        "cache_config_block_size": 48,
        "cache_config_num_gpu_blocks": 329,
        "engine_core_client_type": "FakeEngineCoreClient",
        "engine_core_type": "FakeEngineCore",
        "kv_cache_config_type": "FakeKVCacheConfig",
        "kv_cache_group_count": 1,
        "kv_cache_tensor_count": 1,
        "num_blocks": 329,
        "scheduler_type": "FakeScheduler",
    }
    translated = payload["translated_runtime_cache_config"]
    assert isinstance(translated, dict)
    assert translated["num_blocks"] == 329
    assert translated["group_count"] == 1
    json.dumps(payload, sort_keys=True)


def test_observer_reports_missing_llm_engine_as_unsupported() -> None:
    observed = observe_vllm_runtime_cache_plan(object())

    assert observed.status == "unsupported"
    assert observed.translated_cache_config is None
    assert observed.runtime_path is None
    assert observed.unsupported_reason == "missing runtime attribute `LLM.llm_engine`"
    object_access = cast(dict[str, object], observed.to_dict()["object_access"])
    assert object_access["runtime_resolved_kv_cache_config"] is False


def test_observer_reports_missing_kv_cache_config_as_unsupported() -> None:
    llm = type(
        "PartialLLM",
        (),
        {
            "llm_engine": type(
                "LLMEngine",
                (),
                {
                    "engine_core": type(
                        "EngineCoreClient",
                        (),
                        {
                            "engine_core": type(
                                "EngineCore",
                                (),
                                {"scheduler": object()},
                            )()
                        },
                    )()
                },
            )()
        },
    )()

    observed = observe_vllm_runtime_cache_plan(llm)

    assert observed.status == "unsupported"
    assert observed.unsupported_reason == (
        f"missing runtime attribute `{RUNTIME_KV_CACHE_CONFIG_PATH}`"
    )
    json.dumps(observed.to_dict(), sort_keys=True)
