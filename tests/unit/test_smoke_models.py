"""Smoke test for the models submodule public surface."""

from __future__ import annotations

from cachepawl.models import (
    HYMBA_REF,
    JAMBA_1_5_MINI_REF,
    JAMBA_REF,
    MAMBA2_1B3_REF,
    MAMBA2_REF,
    RECURRENT_GEMMA_REF,
    SAMBA_REF,
    ZAMBA2_REF,
    AttentionLayerProfile,
    HybridModelSpec,
    LayerKind,
    LayerSpec,
    SSMLayerProfile,
)
from cachepawl.quant.dtypes import DType


def test_layer_kind_members() -> None:
    assert {k.name for k in LayerKind} == {"ATTENTION", "MAMBA1", "MAMBA2", "MOE_ATTENTION"}


def test_hybrid_model_spec_dataclass() -> None:
    spec = HybridModelSpec(
        name="toy",
        layers=(
            LayerSpec(index=0, kind=LayerKind.ATTENTION),
            LayerSpec(index=1, kind=LayerKind.MAMBA2),
        ),
        attention_to_ssm_ratio=0.5,
        attention_profile=AttentionLayerProfile(num_kv_heads=2, head_dim=64),
        ssm_profile=SSMLayerProfile(d_inner=128, d_state=16),
        dtype=DType.BF16,
    )
    assert spec.name == "toy"
    assert len(spec.layers) == 2


def test_placeholder_refs_remain_none() -> None:
    refs = (MAMBA2_REF, JAMBA_REF, ZAMBA2_REF, SAMBA_REF, HYMBA_REF, RECURRENT_GEMMA_REF)
    assert all(ref is None for ref in refs)


def test_jamba_1_5_mini_ref_matches_published_dims() -> None:
    assert JAMBA_1_5_MINI_REF.name == "jamba-1.5-mini"
    assert len(JAMBA_1_5_MINI_REF.layers) == 32
    attn_layers = [
        layer for layer in JAMBA_1_5_MINI_REF.layers if layer.kind is LayerKind.ATTENTION
    ]
    ssm_layers = [
        layer for layer in JAMBA_1_5_MINI_REF.layers if layer.kind is LayerKind.MAMBA2
    ]
    assert len(attn_layers) == 4
    assert len(ssm_layers) == 28
    assert JAMBA_1_5_MINI_REF.attention_profile.num_kv_heads == 8
    assert JAMBA_1_5_MINI_REF.attention_profile.head_dim == 128
    assert JAMBA_1_5_MINI_REF.ssm_profile.d_inner == 8192
    assert JAMBA_1_5_MINI_REF.ssm_profile.d_state == 16


def test_mamba2_1b3_ref_uses_larger_synthetic_state() -> None:
    assert MAMBA2_1B3_REF.name == "mamba2-1b3-synthetic"
    assert MAMBA2_1B3_REF.ssm_profile.d_state == 128
    assert MAMBA2_1B3_REF.attention_profile == JAMBA_1_5_MINI_REF.attention_profile
