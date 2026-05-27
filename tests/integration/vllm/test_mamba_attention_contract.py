"""Tests for read-only vLLM Mamba/attention contract observation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import cast

from cachepawl.integrations.vllm import observe_vllm_mamba_attention_contract


@dataclass(slots=True)
class FakeTensor:
    shape: tuple[int, ...]
    dtype: str
    device: str = "cuda:0"
    layout: str = "torch.strided"

    def stride(self) -> tuple[int, ...]:
        return (4, 1)


@dataclass(slots=True)
class FakeStagedWriteTensor:
    gpu: FakeTensor


@dataclass(slots=True)
class FakeBlockTables:
    block_tables: list[FakeStagedWriteTensor]
    input_block_tables: list[FakeTensor]
    slot_mappings: FakeTensor


@dataclass(slots=True)
class FakeBuilder:
    block_size: int = 16


@dataclass(slots=True)
class FakeAttentionGroup:
    metadata_builders: list[FakeBuilder]
    prefix: str = "model.layers.0.self_attn"


@dataclass(slots=True)
class FakeModelRunner:
    block_tables: FakeBlockTables
    attn_groups: list[list[FakeAttentionGroup]]
    mamba_state_idx: dict[str, int] = field(default_factory=dict)
    state_indices_tensor_d: FakeTensor | None = None


@dataclass(slots=True)
class FakeModelExecutor:
    model_runner: FakeModelRunner


@dataclass(slots=True)
class FakeScheduler:
    pass


@dataclass(slots=True)
class FakeEngineCore:
    scheduler: FakeScheduler
    model_executor: FakeModelExecutor


@dataclass(slots=True)
class FakeEngineCoreClient:
    engine_core: FakeEngineCore


@dataclass(slots=True)
class FakeLLMEngine:
    engine_core: FakeEngineCoreClient

    def step(self) -> list[object]:
        runner = self.engine_core.engine_core.model_executor.model_runner
        runner.mamba_state_idx["0"] = 7
        return []


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
        assert sampling_params is not None
        assert use_tqdm is False
        return ["0"]

    def wait_for_completion(self, *, use_tqdm: bool) -> list[object]:
        assert use_tqdm is False
        return []


def test_mamba_attention_contract_observes_state_idx_and_block_tables() -> None:
    runner = FakeModelRunner(
        block_tables=FakeBlockTables(
            block_tables=[FakeStagedWriteTensor(gpu=FakeTensor(shape=(4, 8), dtype="torch.int32"))],
            input_block_tables=[FakeTensor(shape=(4, 8), dtype="torch.int32")],
            slot_mappings=FakeTensor(shape=(1, 8), dtype="torch.int64"),
        ),
        attn_groups=[[FakeAttentionGroup(metadata_builders=[FakeBuilder()])]],
        state_indices_tensor_d=FakeTensor(shape=(4,), dtype="torch.int32"),
    )
    llm = FakeLLM(
        llm_engine=FakeLLMEngine(
            engine_core=FakeEngineCoreClient(
                engine_core=FakeEngineCore(
                    scheduler=FakeScheduler(),
                    model_executor=FakeModelExecutor(model_runner=runner),
                )
            )
        )
    )

    observed = observe_vllm_mamba_attention_contract(
        llm,
        prompt="Count from one to four:",
        sampling_params=object(),
        max_new_tokens=8,
    )
    payload = observed.to_dict()
    fields = cast(tuple[dict[str, object], ...], payload["fields"])
    snapshots = cast(tuple[dict[str, object], ...], payload["snapshots"])

    assert observed.status == "mamba_attention_contract_observation"
    assert observed.request_id == "0"
    assert {field["name"]: field["status"] for field in fields} == {
        "mamba_state_index_contract": "observed",
        "attention_block_table_view_contract": "observed",
        "attention_metadata_builder_contract": "observed",
        "mamba_state_tensor_contract": "observed",
    }
    after_first_step = next(
        snapshot for snapshot in snapshots if snapshot["phase"] == "after_first_step"
    )
    totals = cast(dict[str, object], after_first_step["totals"])
    block_table_tensor_count = totals["block_table_tensor_count"]
    assert totals["mamba_state_idx_contains_request"] is True
    assert isinstance(block_table_tensor_count, int)
    assert block_table_tensor_count >= 2
    json.dumps(payload, sort_keys=True)


def test_mamba_attention_contract_blocks_when_enqueue_missing() -> None:
    llm = type(
        "PartialLLM",
        (),
        {
            "llm_engine": FakeLLMEngine(
                engine_core=FakeEngineCoreClient(
                    engine_core=FakeEngineCore(
                        scheduler=FakeScheduler(),
                        model_executor=FakeModelExecutor(
                            model_runner=FakeModelRunner(
                                block_tables=FakeBlockTables([], [], FakeTensor((1,), "int")),
                                attn_groups=[],
                            )
                        ),
                    )
                )
            )
        },
    )()

    observed = observe_vllm_mamba_attention_contract(
        llm,
        prompt="Count from one to four:",
        sampling_params=object(),
        max_new_tokens=8,
    )

    assert observed.status == "unsupported"
    assert observed.unsupported_reason == "LLM.enqueue was not reachable"
