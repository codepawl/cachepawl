"""Smoke test for the models submodule public surface."""

from __future__ import annotations

from cachepawl.models import (
    HYMBA_REF,
    JAMBA_REF,
    MAMBA2_REF,
    RECURRENT_GEMMA_REF,
    SAMBA_REF,
    ZAMBA2_REF,
    HybridModelSpec,
    LayerKind,
    LayerSpec,
)


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
    )
    assert spec.name == "toy"
    assert len(spec.layers) == 2


def test_reference_configs_are_placeholders() -> None:
    refs = (MAMBA2_REF, JAMBA_REF, ZAMBA2_REF, SAMBA_REF, HYMBA_REF, RECURRENT_GEMMA_REF)
    assert all(ref is None for ref in refs)
