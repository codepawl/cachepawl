"""Tests for read-only vLLM runtime contract observation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from cachepawl.integrations.vllm import (
    RUNTIME_CONTRACT_BASE_PATH,
    observe_vllm_runtime_contracts,
)


@dataclass(frozen=True, slots=True)
class FakeTensor:
    shape: tuple[int, ...]
    dtype: str
    device: str = "cuda:0"
    layout: str = "torch.strided"

    def stride(self) -> tuple[int, ...]:
        return (12, 4, 1)


@dataclass(frozen=True, slots=True)
class FakeBlockPool:
    num_gpu_blocks: int

    def get_num_free_blocks(self) -> int:
        return 321


@dataclass(frozen=True, slots=True)
class FakeCoordinator:
    block_pool: FakeBlockPool


@dataclass(frozen=True, slots=True)
class FakeKVCacheConfig:
    num_blocks: int


@dataclass(frozen=True, slots=True)
class FakeKVCacheManager:
    kv_cache_config: FakeKVCacheConfig
    coordinator: FakeCoordinator
    block_pool: FakeBlockPool

    def allocate_slots(self) -> None:
        raise AssertionError("must not be called")

    def free(self) -> None:
        raise AssertionError("must not be called")

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        raise AssertionError(f"must not be called for {request_id}")

    def get_usage(self) -> float:
        return 0.125

    def take_events(self) -> None:
        raise AssertionError("must not be called")

    def take_new_block_ids(self) -> None:
        raise AssertionError("must not be called")


@dataclass(frozen=True, slots=True)
class FakeScheduler:
    kv_cache_config: FakeKVCacheConfig
    kv_cache_manager: FakeKVCacheManager


@dataclass(frozen=True, slots=True)
class FakeModelRunner:
    kv_caches: dict[str, FakeTensor]


@dataclass(frozen=True, slots=True)
class FakeDriverWorker:
    model_runner: FakeModelRunner


@dataclass(frozen=True, slots=True)
class FakeModelExecutor:
    driver_worker: FakeDriverWorker


@dataclass(frozen=True, slots=True)
class FakeEngineCore:
    scheduler: FakeScheduler
    model_executor: FakeModelExecutor


@dataclass(frozen=True, slots=True)
class FakeEngineCoreClient:
    engine_core: FakeEngineCore


@dataclass(frozen=True, slots=True)
class FakeLLMEngine:
    engine_core: FakeEngineCoreClient


@dataclass(frozen=True, slots=True)
class FakeLLM:
    llm_engine: FakeLLMEngine


def test_runtime_contract_observes_manager_usage_and_worker_tensors() -> None:
    config = FakeKVCacheConfig(num_blocks=329)
    block_pool = FakeBlockPool(num_gpu_blocks=329)
    manager = FakeKVCacheManager(
        kv_cache_config=config,
        coordinator=FakeCoordinator(block_pool=block_pool),
        block_pool=block_pool,
    )
    llm = FakeLLM(
        llm_engine=FakeLLMEngine(
            engine_core=FakeEngineCoreClient(
                engine_core=FakeEngineCore(
                    scheduler=FakeScheduler(
                        kv_cache_config=config,
                        kv_cache_manager=manager,
                    ),
                    model_executor=FakeModelExecutor(
                        driver_worker=FakeDriverWorker(
                            model_runner=FakeModelRunner(
                                kv_caches={
                                    "layer.0": FakeTensor(
                                        shape=(2, 3, 4),
                                        dtype="torch.bfloat16",
                                    )
                                }
                            )
                        )
                    ),
                )
            )
        )
    )

    observed = observe_vllm_runtime_contracts(llm)
    payload = observed.to_dict()
    scheduler_manager = cast(dict[str, object], payload["scheduler_manager"])
    block_usage = cast(dict[str, object], payload["block_usage"])
    worker_tensors = cast(tuple[dict[str, object], ...], payload["worker_tensors"])
    blockers = cast(tuple[dict[str, object], ...], payload["field_level_blockers"])

    assert observed.status == "runtime_contract_observation"
    assert observed.runtime_path == RUNTIME_CONTRACT_BASE_PATH
    assert scheduler_manager["kv_cache_manager_type"] == "FakeKVCacheManager"
    assert scheduler_manager["manager_config_matches_scheduler"] is True
    assert block_usage["manager_get_usage"] == 0.125
    assert block_usage["block_pool_free_blocks"] == 321
    assert block_usage["block_pool_num_gpu_blocks"] == 329
    assert worker_tensors[0]["shape"] == (2, 3, 4)
    assert worker_tensors[0]["stride"] == (12, 4, 1)
    assert worker_tensors[0]["dtype"] == "torch.bfloat16"
    assert {blocker["name"] for blocker in blockers} == {
        "request_to_block_assignment",
        "mamba_state_index_attention_view_contract",
    }
    json.dumps(payload, sort_keys=True)


def test_runtime_contract_reports_missing_scheduler_as_unsupported() -> None:
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
                        {"engine_core": object()},
                    )()
                },
            )()
        },
    )()

    observed = observe_vllm_runtime_contracts(llm)

    assert observed.status == "unsupported"
    assert observed.unsupported_reason == (
        f"missing runtime attribute `{RUNTIME_CONTRACT_BASE_PATH}.scheduler`"
    )
    object_access = cast(dict[str, object], observed.to_dict()["object_access"])
    assert object_access["runtime_contract_objects_reached"] is False
