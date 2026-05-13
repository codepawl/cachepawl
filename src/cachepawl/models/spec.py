"""Hybrid model layout specifications.

Layer geometry types (``AttentionLayerProfile``, ``SSMLayerProfile``)
live here so allocators and workload generators share one source of
truth; ``cachepawl.benchmarks.harness.workloads`` re-exports them.

Reference constants:

- ``JAMBA_1_5_MINI_REF`` is the real Jamba-1.5-Mini layout, dims sourced
  from the HuggingFace ``transformers.JambaConfig`` defaults.
- ``MAMBA2_1B3_REF`` is **synthetic**: Jamba-style layer counts paired
  with the larger Mamba-2 1.3B SSM state size (``d_state = 128``). It
  exists because the real Mamba-2 1.3B has zero attention layers, so a
  pure Mamba-2 spec cannot exercise the cross-cache padding-waste
  pathology that the ``PaddedUnifiedPool`` baseline must surface.

Other ``*_REF`` constants stay ``None`` until an upstream config is
mapped in.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final

from cachepawl.quant.dtypes import DType


class LayerKind(enum.Enum):
    """Layer kinds spanning attention, SSM, and routed variants."""

    ATTENTION = "attention"
    MAMBA1 = "mamba1"
    MAMBA2 = "mamba2"
    MOE_ATTENTION = "moe_attention"


@dataclass(frozen=True, slots=True)
class AttentionLayerProfile:
    """KV-side shape parameters for one attention layer."""

    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True, slots=True)
class SSMLayerProfile:
    """State-side shape parameters for one SSM layer."""

    d_inner: int
    d_state: int


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """One layer entry inside a hybrid model layout."""

    index: int
    kind: LayerKind


@dataclass(frozen=True, slots=True)
class HybridModelSpec:
    """High-level description of a hybrid Mamba-Transformer-MoE model.

    Carries enough information for a cache allocator to compute per-layer
    page sizes without consulting any external config.
    """

    name: str
    layers: tuple[LayerSpec, ...]
    attention_to_ssm_ratio: float
    attention_profile: AttentionLayerProfile
    ssm_profile: SSMLayerProfile
    dtype: DType


def _jamba_mini_layer_pattern(
    total_layers: int = 32, attn_every: int = 8
) -> tuple[LayerSpec, ...]:
    """Build a Jamba-Mini-style 1:7 attention:mamba interleave.

    Layer index ``i`` is attention when ``i % attn_every == 0`` and
    Mamba-2 otherwise. The default 32 layers with ``attn_every=8``
    yields 4 attention layers and 28 Mamba layers.
    """

    layers: list[LayerSpec] = []
    for i in range(total_layers):
        kind = LayerKind.ATTENTION if i % attn_every == 0 else LayerKind.MAMBA2
        layers.append(LayerSpec(index=i, kind=kind))
    return tuple(layers)


JAMBA_1_5_MINI_REF: Final[HybridModelSpec] = HybridModelSpec(
    name="jamba-1.5-mini",
    layers=_jamba_mini_layer_pattern(32, attn_every=8),
    attention_to_ssm_ratio=4.0 / 28.0,
    attention_profile=AttentionLayerProfile(num_kv_heads=8, head_dim=128),
    ssm_profile=SSMLayerProfile(d_inner=8192, d_state=16),
    dtype=DType.BF16,
)

MAMBA2_1B3_REF: Final[HybridModelSpec] = HybridModelSpec(
    name="mamba2-1b3-synthetic",
    layers=_jamba_mini_layer_pattern(32, attn_every=8),
    attention_to_ssm_ratio=4.0 / 28.0,
    attention_profile=AttentionLayerProfile(num_kv_heads=8, head_dim=128),
    ssm_profile=SSMLayerProfile(d_inner=8192, d_state=128),
    dtype=DType.BF16,
)

# Real-config placeholders. Stay None until each upstream config is mapped in.
MAMBA2_REF: Final[HybridModelSpec | None] = None
JAMBA_REF: Final[HybridModelSpec | None] = None
ZAMBA2_REF: Final[HybridModelSpec | None] = None
SAMBA_REF: Final[HybridModelSpec | None] = None
HYMBA_REF: Final[HybridModelSpec | None] = None
RECURRENT_GEMMA_REF: Final[HybridModelSpec | None] = None
