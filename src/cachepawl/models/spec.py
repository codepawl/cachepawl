"""Hybrid model layout specifications.

Each reference config below is left as ``None`` until the upstream
architecture is wired up. They are typed as ``Final`` so downstream
code can narrow with a simple ``is not None`` check.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final


class LayerKind(enum.Enum):
    """Layer kinds spanning attention, SSM, and routed variants."""

    ATTENTION = "attention"
    MAMBA1 = "mamba1"
    MAMBA2 = "mamba2"
    MOE_ATTENTION = "moe_attention"


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """One layer entry inside a hybrid model layout."""

    index: int
    kind: LayerKind


@dataclass(frozen=True, slots=True)
class HybridModelSpec:
    """High-level description of a hybrid Mamba-Transformer-MoE model."""

    name: str
    layers: tuple[LayerSpec, ...]
    attention_to_ssm_ratio: float


# TODO(spec): fill from upstream config
MAMBA2_REF: Final[HybridModelSpec | None] = None
# TODO(spec): fill from upstream config
JAMBA_REF: Final[HybridModelSpec | None] = None
# TODO(spec): fill from upstream config
ZAMBA2_REF: Final[HybridModelSpec | None] = None
# TODO(spec): fill from upstream config
SAMBA_REF: Final[HybridModelSpec | None] = None
# TODO(spec): fill from upstream config
HYMBA_REF: Final[HybridModelSpec | None] = None
# TODO(spec): fill from upstream config
RECURRENT_GEMMA_REF: Final[HybridModelSpec | None] = None
