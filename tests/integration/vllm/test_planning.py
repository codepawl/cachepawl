"""Tests for the import-safe vLLM integration skeleton."""

from __future__ import annotations

import dataclasses

import pytest

import cachepawl.integrations.vllm as vllm_integration
from cachepawl.integrations.vllm import (
    VllmCacheLayerSpec,
    VllmIntegrationPlan,
    get_vllm_version,
    is_vllm_available,
    plan_vllm_integration,
    translate_hybrid_model_spec,
)
from cachepawl.models.spec import JAMBA_1_5_MINI_REF, LayerKind
from cachepawl.quant.dtypes import DType


def test_vllm_package_imports_without_vllm_dependency() -> None:
    assert vllm_integration.__all__ == [
        "VllmCacheLayerSpec",
        "VllmCachePlan",
        "VllmIntegrationPlan",
        "VllmTranslatedCacheConfig",
        "VllmTranslatedCacheGroup",
        "VllmTranslatedCacheSpec",
        "VllmTranslatedCacheTensor",
        "VllmTranslationError",
        "get_vllm_version",
        "is_vllm_available",
        "plan_vllm_integration",
        "translate_hybrid_model_spec",
        "translate_kv_cache_config",
        "translate_kv_cache_group",
        "translate_kv_cache_spec",
        "translate_kv_cache_tensor",
    ]


def test_availability_helpers_are_graceful() -> None:
    available = is_vllm_available()
    version = get_vllm_version()
    assert isinstance(available, bool)
    if not available:
        assert version is None


def test_layer_spec_is_frozen_slots_value_record() -> None:
    layer = VllmCacheLayerSpec(
        layer_index=0,
        cache_kind="attention",
        source_kind=LayerKind.ATTENTION,
        dtype=DType.BF16,
        num_kv_heads=8,
        head_dim=128,
    )
    assert layer == VllmCacheLayerSpec(
        layer_index=0,
        cache_kind="attention",
        source_kind=LayerKind.ATTENTION,
        dtype=DType.BF16,
        num_kv_heads=8,
        head_dim=128,
    )
    assert not hasattr(layer, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        layer.layer_index = 1  # type: ignore[misc]


def test_translate_jamba_ref_to_structural_cache_plan() -> None:
    plan = translate_hybrid_model_spec(JAMBA_1_5_MINI_REF, block_size_tokens=16)
    assert plan.model_name == "jamba-1.5-mini"
    assert plan.dtype is DType.BF16
    assert plan.block_size_tokens == 16
    assert len(plan.layers) == 32
    assert plan.attention_layers == 4
    assert plan.ssm_layers == 28

    first_attention = plan.layers[0]
    assert first_attention.cache_kind == "attention"
    assert first_attention.source_kind is LayerKind.ATTENTION
    assert first_attention.num_kv_heads == 8
    assert first_attention.head_dim == 128
    assert first_attention.d_inner is None
    assert first_attention.d_state is None

    first_ssm = plan.layers[1]
    assert first_ssm.cache_kind == "ssm"
    assert first_ssm.source_kind is LayerKind.MAMBA2
    assert first_ssm.d_inner == 8192
    assert first_ssm.d_state == 16
    assert first_ssm.num_kv_heads is None
    assert first_ssm.head_dim is None


def test_translate_rejects_nonpositive_block_size() -> None:
    with pytest.raises(ValueError, match="block_size_tokens"):
        translate_hybrid_model_spec(JAMBA_1_5_MINI_REF, block_size_tokens=0)


def test_plan_vllm_integration_is_runtime_safe_without_spec() -> None:
    plan = plan_vllm_integration()
    assert isinstance(plan, VllmIntegrationPlan)
    assert plan.runtime_ready is False
    assert plan.cache_plan is None
    assert "runtime shim not implemented" in plan.notes


def test_plan_vllm_integration_accepts_model_spec() -> None:
    plan = plan_vllm_integration(JAMBA_1_5_MINI_REF)
    assert plan.cache_plan is not None
    assert plan.cache_plan.attention_layers == 4
    assert plan.cache_plan.ssm_layers == 28
