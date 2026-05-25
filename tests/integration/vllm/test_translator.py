"""Tests for duck-typed vLLM cache-plan translation."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

import cachepawl.integrations.vllm as vllm_integration
from cachepawl.integrations.vllm import (
    VllmTranslationError,
    translate_kv_cache_config,
    translate_kv_cache_group,
    translate_kv_cache_spec,
    translate_kv_cache_tensor,
)


@dataclass(frozen=True, slots=True)
class FakeAttentionSpec:
    block_size: int
    page_size_bytes: int
    real_page_size_bytes: int
    num_kv_heads: int
    head_size: int
    dtype: str
    kv_quant_mode: str | None = None
    page_size_padded: int | None = None


@dataclass(frozen=True, slots=True)
class FakeMambaSpec:
    block_size: int
    page_size_bytes: int
    shapes: dict[str, tuple[int, ...]]
    dtypes: dict[str, str]
    mamba_type: str
    mamba_cache_mode: str
    num_speculative_blocks: int = 0
    page_size_padded: int | None = None


@dataclass(frozen=True, slots=True)
class FakeRealMambaSpec:
    block_size: int
    page_size_bytes: int
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    mamba_cache_mode: str


@dataclass(frozen=True, slots=True)
class FakeKVCacheGroupSpec:
    layer_names: tuple[str, ...]
    kv_cache_spec: object
    group_id: str | None = None
    is_eagle_group: bool = False


@dataclass(frozen=True, slots=True)
class FakeKVCacheTensor:
    size: int
    shared_by: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FakeKVCacheConfig:
    num_blocks: int
    kv_cache_groups: tuple[FakeKVCacheGroupSpec, ...]
    kv_cache_tensors: tuple[FakeKVCacheTensor, ...]
    block_size: int = 16
    cache_dtype: str = "bfloat16"


def test_package_exports_translator_without_vllm_dependency() -> None:
    assert "translate_kv_cache_config" in vllm_integration.__all__


def test_translate_attention_spec_extracts_stable_planner_fields() -> None:
    translated = translate_kv_cache_spec(
        "layers.0.self_attn",
        FakeAttentionSpec(
            block_size=16,
            page_size_bytes=65_536,
            real_page_size_bytes=65_536,
            num_kv_heads=8,
            head_size=128,
            dtype="bfloat16",
        ),
    )

    assert translated.cache_kind == "attention"
    assert translated.layer_name == "layers.0.self_attn"
    assert translated.block_size == 16
    assert translated.page_size_bytes == 65_536
    assert translated.useful_bytes == 65_536
    assert translated.dtype == "bfloat16"
    assert translated.metadata["num_kv_heads"] == 8
    assert translated.metadata["head_size"] == 128
    json.dumps(translated.to_dict(), sort_keys=True)


def test_translate_mamba_spec_derives_useful_bytes_from_shapes() -> None:
    translated = translate_kv_cache_spec(
        "layers.1.mamba",
        FakeMambaSpec(
            block_size=1,
            page_size_bytes=262_144,
            shapes={
                "conv_state": (4, 64),
                "ssm_state": (8192, 16),
            },
            dtypes={
                "conv_state": "float16",
                "ssm_state": "bfloat16",
            },
            mamba_type="mamba2",
            mamba_cache_mode="align",
            page_size_padded=64,
        ),
    )

    assert translated.cache_kind == "mamba"
    assert translated.page_size_bytes == 262_144
    assert translated.useful_bytes == (4 * 64 * 2) + (8192 * 16 * 2)
    assert translated.dtype is None
    assert translated.metadata["mamba_cache_mode"] == "align"
    assert translated.metadata["dtypes"] == {
        "conv_state": "float16",
        "ssm_state": "bfloat16",
    }
    json.dumps(translated.to_dict(), sort_keys=True)


def test_translate_real_mamba_shape_dtype_tuples() -> None:
    translated = translate_kv_cache_spec(
        "layers.1.mamba",
        FakeRealMambaSpec(
            block_size=1,
            page_size_bytes=262_144,
            shapes=((8192, 16),),
            dtypes=("torch.bfloat16",),
            mamba_cache_mode="align",
        ),
    )

    assert translated.cache_kind == "mamba"
    assert translated.useful_bytes == 8192 * 16 * 2
    assert "max_memory_usage_bytes" not in translated.metadata
    json.dumps(translated.to_dict(), sort_keys=True)


def test_translate_hybrid_cache_config_is_deterministic() -> None:
    config = FakeKVCacheConfig(
        num_blocks=32,
        kv_cache_groups=(
            FakeKVCacheGroupSpec(
                layer_names=("layers.0.self_attn", "layers.8.self_attn"),
                kv_cache_spec=FakeAttentionSpec(
                    block_size=16,
                    page_size_bytes=65_536,
                    real_page_size_bytes=65_536,
                    num_kv_heads=8,
                    head_size=128,
                    dtype="bfloat16",
                ),
                group_id="attention",
            ),
            FakeKVCacheGroupSpec(
                layer_names=("layers.1.mamba",),
                kv_cache_spec=FakeMambaSpec(
                    block_size=1,
                    page_size_bytes=262_144,
                    shapes={"ssm_state": (8192, 16)},
                    dtypes={"ssm_state": "bfloat16"},
                    mamba_type="mamba2",
                    mamba_cache_mode="align",
                ),
                group_id="mamba",
            ),
        ),
        kv_cache_tensors=(
            FakeKVCacheTensor(size=2_097_152, shared_by=("layers.0.self_attn",)),
            FakeKVCacheTensor(size=8_388_608, shared_by=("layers.1.mamba",)),
        ),
    )

    translated = translate_kv_cache_config(config)

    assert translated.num_blocks == 32
    assert translated.group_count == 2
    assert translated.layer_count == 3
    assert translated.attention_group_count == 1
    assert translated.mamba_group_count == 1
    assert translated.total_page_size_bytes == 327_680
    assert translated.total_useful_bytes == 327_680
    assert [group.group_name for group in translated.groups] == ["attention", "mamba"]
    assert [tensor.size_bytes for tensor in translated.tensors] == [2_097_152, 8_388_608]
    assert translated.to_dict() == translate_kv_cache_config(config).to_dict()
    json.dumps(translated.to_dict(), sort_keys=True)


def test_translate_group_accepts_layer_group_name_fallback() -> None:
    group = type(
        "LayerGroup",
        (),
        {
            "layers": ("layer.0",),
            "spec": FakeAttentionSpec(
                block_size=16,
                page_size_bytes=128,
                real_page_size_bytes=128,
                num_kv_heads=1,
                head_size=4,
                dtype="float16",
            ),
            "layer_group_name": "attn-group",
        },
    )()

    translated = translate_kv_cache_group(0, group)
    assert translated.group_name == "attn-group"
    assert translated.layer_names == ("layer.0",)


def test_translate_tensor_accepts_size_bytes_alias() -> None:
    tensor = type("Tensor", (), {"size_bytes": 1024, "shared_by": ["a", "b"]})()
    translated = translate_kv_cache_tensor(tensor)
    assert translated.size_bytes == 1024
    assert translated.shared_by == ("a", "b")


def test_unsupported_object_raises_typed_error_not_attribute_error() -> None:
    with pytest.raises(VllmTranslationError, match="unsupported vLLM cache spec"):
        translate_kv_cache_spec("layer.unknown", object())
