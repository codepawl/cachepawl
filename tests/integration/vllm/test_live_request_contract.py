"""Tests for read-only vLLM live-request contract observation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import cast

from cachepawl.integrations.vllm import observe_vllm_live_request_contract


@dataclass(slots=True)
class FakeBlockPool:
    free_blocks: int = 10
    num_gpu_blocks: int = 12

    def get_num_free_blocks(self) -> int:
        return self.free_blocks


@dataclass(slots=True)
class FakeRequest:
    request_id: str


@dataclass(slots=True)
class FakeKVCacheManager:
    block_pool: FakeBlockPool
    blocks: dict[str, tuple[list[int], ...]] = field(default_factory=dict)

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        return self.blocks.get(request_id, ([],))


@dataclass(slots=True)
class FakeScheduler:
    kv_cache_manager: FakeKVCacheManager
    requests: dict[str, FakeRequest] = field(default_factory=dict)
    running: list[FakeRequest] = field(default_factory=list)
    waiting: list[FakeRequest] = field(default_factory=list)
    skipped_waiting: list[FakeRequest] = field(default_factory=list)
    prev_step_scheduled_req_ids: set[str] = field(default_factory=set)
    finished_req_ids: set[str] = field(default_factory=set)


@dataclass(slots=True)
class FakeEngineCore:
    scheduler: FakeScheduler


@dataclass(slots=True)
class FakeEngineCoreClient:
    engine_core: FakeEngineCore


@dataclass(slots=True)
class FakeStepOutput:
    request_id: str
    finished: bool


@dataclass(slots=True)
class FakeCompletion:
    token_ids: tuple[int, ...]


@dataclass(slots=True)
class FakeRequestOutput:
    request_id: str
    finished: bool
    outputs: tuple[FakeCompletion, ...]


@dataclass(slots=True)
class FakeLLMEngine:
    engine_core: FakeEngineCoreClient
    unfinished: int = 0

    def get_num_unfinished_requests(self) -> int:
        return self.unfinished

    def has_unfinished_requests(self) -> bool:
        return self.unfinished > 0

    def step(self) -> list[FakeStepOutput]:
        scheduler = self.engine_core.engine_core.scheduler
        request = scheduler.waiting.pop()
        scheduler.running.append(request)
        scheduler.prev_step_scheduled_req_ids.add(request.request_id)
        scheduler.kv_cache_manager.blocks[request.request_id] = ([3, 4],)
        scheduler.kv_cache_manager.block_pool.free_blocks = 8
        return [FakeStepOutput(request_id=request.request_id, finished=False)]


@dataclass(slots=True)
class FakeLLM:
    llm_engine: FakeLLMEngine

    def enqueue(
        self,
        prompts: list[str],
        sampling_params: object,
        *,
        use_tqdm: bool,
    ) -> list[str]:
        assert prompts == ["Count from one to four:"]
        assert sampling_params == object() or sampling_params is not None
        assert use_tqdm is False
        request = FakeRequest(request_id="0")
        scheduler = self.llm_engine.engine_core.engine_core.scheduler
        scheduler.requests[request.request_id] = request
        scheduler.waiting.append(request)
        self.llm_engine.unfinished = 1
        return [request.request_id]

    def wait_for_completion(self, *, use_tqdm: bool) -> list[FakeRequestOutput]:
        assert use_tqdm is False
        scheduler = self.llm_engine.engine_core.engine_core.scheduler
        request = scheduler.running.pop()
        scheduler.finished_req_ids.add(request.request_id)
        scheduler.requests.pop(request.request_id)
        scheduler.kv_cache_manager.blocks.pop(request.request_id)
        scheduler.kv_cache_manager.block_pool.free_blocks = 10
        self.llm_engine.unfinished = 0
        return [
            FakeRequestOutput(
                request_id=request.request_id,
                finished=True,
                outputs=(FakeCompletion(token_ids=(11, 12, 13)),),
            )
        ]


def test_live_request_contract_observes_active_block_assignment() -> None:
    block_pool = FakeBlockPool()
    scheduler = FakeScheduler(kv_cache_manager=FakeKVCacheManager(block_pool=block_pool))
    llm = FakeLLM(
        llm_engine=FakeLLMEngine(
            engine_core=FakeEngineCoreClient(engine_core=FakeEngineCore(scheduler=scheduler))
        )
    )

    observed = observe_vllm_live_request_contract(
        llm,
        prompt="Count from one to four:",
        sampling_params=object(),
        max_new_tokens=8,
    )
    payload = observed.to_dict()
    fields = cast(tuple[dict[str, object], ...], payload["fields"])
    snapshots = cast(tuple[dict[str, object], ...], payload["snapshots"])

    assert observed.status == "live_request_contract_observation"
    assert observed.request_id == "0"
    assert {field["name"]: field["status"] for field in fields} == {
        "live_request_id": "observed",
        "request_to_block_assignment": "observed",
        "scheduler_request_metadata": "observed",
        "before_after_block_pool_usage": "observed",
    }
    after_first_step = next(
        snapshot for snapshot in snapshots if snapshot["phase"] == "after_first_step"
    )
    request_block_ids = cast(dict[str, object], after_first_step["request_block_ids"])
    assert request_block_ids["total_block_id_count"] == 2
    assert request_block_ids["sample_block_ids"] == (3, 4)
    after_completion = next(
        snapshot for snapshot in snapshots if snapshot["phase"] == "after_completion"
    )
    after_blocks = cast(dict[str, object], after_completion["request_block_ids"])
    assert after_blocks["total_block_id_count"] == 0
    json.dumps(payload, sort_keys=True)


def test_live_request_contract_blocks_when_enqueue_missing() -> None:
    scheduler = FakeScheduler(kv_cache_manager=FakeKVCacheManager(block_pool=FakeBlockPool()))
    llm = type(
        "PartialLLM",
        (),
        {
            "llm_engine": FakeLLMEngine(
                engine_core=FakeEngineCoreClient(engine_core=FakeEngineCore(scheduler=scheduler))
            )
        },
    )()

    observed = observe_vllm_live_request_contract(
        llm,
        prompt="Count from one to four:",
        sampling_params=object(),
        max_new_tokens=8,
    )

    assert observed.status == "unsupported"
    assert observed.unsupported_reason == "LLM.enqueue was not reachable"
