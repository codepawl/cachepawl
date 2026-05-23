"""Typed vLLM integration planning records.

The records in this module are deliberately structural and vLLM-free.
They let tests and future integration work agree on the cache shape
without making ``vllm`` an install-time dependency of cachepawl.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from importlib.util import find_spec
from typing import Literal, TypeAlias

from cachepawl.models.spec import HybridModelSpec, LayerKind
from cachepawl.quant.dtypes import DType

VllmCacheKind: TypeAlias = Literal["attention", "ssm"]


@dataclass(frozen=True, slots=True)
class VllmCacheLayerSpec:
    """One cache-bearing layer in the planned vLLM integration."""

    layer_index: int
    cache_kind: VllmCacheKind
    source_kind: LayerKind
    dtype: DType
    num_kv_heads: int | None = None
    head_dim: int | None = None
    d_inner: int | None = None
    d_state: int | None = None


@dataclass(frozen=True, slots=True)
class VllmCachePlan:
    """Structural cache plan derived from a cachepawl hybrid model spec."""

    model_name: str
    layers: tuple[VllmCacheLayerSpec, ...]
    attention_layers: int
    ssm_layers: int
    dtype: DType
    block_size_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class VllmIntegrationPlan:
    """Top-level planning result for the provisional vLLM integration."""

    vllm_available: bool
    vllm_version: str | None
    cache_plan: VllmCachePlan | None
    runtime_ready: bool
    notes: tuple[str, ...]


def is_vllm_available() -> bool:
    """Return whether a ``vllm`` distribution is import-discoverable."""

    return find_spec("vllm") is not None


def get_vllm_version() -> str | None:
    """Return the installed vLLM package version without importing vLLM."""

    try:
        return package_version("vllm")
    except PackageNotFoundError:
        return None


def translate_hybrid_model_spec(
    spec: HybridModelSpec, *, block_size_tokens: int | None = None
) -> VllmCachePlan:
    """Translate a cachepawl hybrid model spec into a structural vLLM plan."""

    if block_size_tokens is not None and block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive when provided")

    layers = tuple(
        _translate_layer(spec, layer_kind.index, layer_kind.kind) for layer_kind in spec.layers
    )
    attention_layers = sum(1 for layer in layers if layer.cache_kind == "attention")
    ssm_layers = sum(1 for layer in layers if layer.cache_kind == "ssm")
    return VllmCachePlan(
        model_name=spec.name,
        layers=layers,
        attention_layers=attention_layers,
        ssm_layers=ssm_layers,
        dtype=spec.dtype,
        block_size_tokens=block_size_tokens,
    )


def plan_vllm_integration(spec: HybridModelSpec | None = None) -> VllmIntegrationPlan:
    """Build a provisional vLLM integration plan without touching vLLM runtime APIs."""

    cache_plan = translate_hybrid_model_spec(spec) if spec is not None else None
    notes = (
        "runtime shim not implemented",
        "vLLM remains an optional external dependency",
        "Path C KVCacheManager subclass work remains pending",
    )
    return VllmIntegrationPlan(
        vllm_available=is_vllm_available(),
        vllm_version=get_vllm_version(),
        cache_plan=cache_plan,
        runtime_ready=False,
        notes=notes,
    )


def _translate_layer(
    spec: HybridModelSpec, layer_index: int, source_kind: LayerKind
) -> VllmCacheLayerSpec:
    if source_kind in (LayerKind.ATTENTION, LayerKind.MOE_ATTENTION):
        return VllmCacheLayerSpec(
            layer_index=layer_index,
            cache_kind="attention",
            source_kind=source_kind,
            dtype=spec.dtype,
            num_kv_heads=spec.attention_profile.num_kv_heads,
            head_dim=spec.attention_profile.head_dim,
        )
    return VllmCacheLayerSpec(
        layer_index=layer_index,
        cache_kind="ssm",
        source_kind=source_kind,
        dtype=spec.dtype,
        d_inner=spec.ssm_profile.d_inner,
        d_state=spec.ssm_profile.d_state,
    )
