"""Benchmark harness public surface and allocator registry.

The registry ships with two baseline allocators (``padded_unified``,
``fixed_dual``) registered at import time. Future allocator PRs call
:func:`register_allocator` to plug in.
"""

from collections.abc import Callable

import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.allocator.base import Allocator
from cachepawl.allocator.baselines import FixedDualPool, PaddedUnifiedPool
from cachepawl.benchmarks.harness.metrics import (
    AllocatorMetrics,
    LatencyPercentiles,
    MetricsCollector,
    compute_percentiles,
)
from cachepawl.benchmarks.harness.runner import run_benchmark
from cachepawl.benchmarks.harness.schema import (
    SCHEMA_VERSION,
    BenchmarkRun,
    Environment,
    Hardware,
)
from cachepawl.benchmarks.harness.workloads import (
    JAMBA_MINI_ATTN,
    JAMBA_MINI_SSM,
    PRESETS,
    Request,
    WorkloadSpec,
    generate_request_stream,
    per_sequence_ssm_bytes,
    per_token_kv_bytes,
)
from cachepawl.models.spec import (
    MAMBA2_1B3_REF,
    AttentionLayerProfile,
    HybridModelSpec,
    LayerKind,
    LayerSpec,
    SSMLayerProfile,
)

AllocatorFactory = Callable[[WorkloadSpec, torch.device], Allocator]
REGISTRY: dict[str, AllocatorFactory] = {}

_DEFAULT_TOTAL_BYTES: int = 8 * 1024**3


def register_allocator(name: str, factory: AllocatorFactory) -> None:
    """Register an allocator factory under ``name``.

    Names are unique. Overwriting an existing entry is permitted so test
    fixtures can swap implementations within one process.
    """

    REGISTRY[name] = factory


def _hybrid_spec_from_workload(spec: WorkloadSpec) -> HybridModelSpec:
    """Build a ``HybridModelSpec`` from the harness's ``WorkloadSpec``.

    The runner's WorkloadSpec carries layer counts and per-kind profiles
    but no explicit layer ordering. The factory synthesizes an
    interleaved pattern (attention every ``attn_every`` layers) that
    matches the Jamba-style layout most baseline tests assume.
    """

    total_layers = spec.attention_layers + spec.ssm_layers
    if spec.attention_layers <= 0:
        attn_every = total_layers + 1
    else:
        attn_every = max(1, total_layers // spec.attention_layers)
    layers: list[LayerSpec] = []
    for i in range(total_layers):
        kind = LayerKind.ATTENTION if i % attn_every == 0 else LayerKind.MAMBA2
        layers.append(LayerSpec(index=i, kind=kind))
    ratio = float(spec.attention_layers) / float(max(1, spec.ssm_layers))
    return HybridModelSpec(
        name=f"workload-{spec.name}",
        layers=tuple(layers),
        attention_to_ssm_ratio=ratio,
        attention_profile=spec.attention_profile,
        ssm_profile=spec.ssm_profile,
        dtype=spec.dtype,
    )


def _padded_unified_factory(spec: WorkloadSpec, device: torch.device) -> PaddedUnifiedPool:
    return PaddedUnifiedPool(
        model_spec=_hybrid_spec_from_workload(spec),
        total_bytes=_DEFAULT_TOTAL_BYTES,
        device=device,
    )


def _fixed_dual_factory(spec: WorkloadSpec, device: torch.device) -> FixedDualPool:
    return FixedDualPool(
        model_spec=_hybrid_spec_from_workload(spec),
        total_bytes=_DEFAULT_TOTAL_BYTES,
        device=device,
    )


def _avmp_static_factory(spec: WorkloadSpec, device: torch.device) -> AsymmetricVirtualPool:
    return AsymmetricVirtualPool(
        model_spec=_hybrid_spec_from_workload(spec),
        total_bytes=_DEFAULT_TOTAL_BYTES,
        device=device,
    )


def _avmp_dynamic_factory(spec: WorkloadSpec, device: torch.device) -> AsymmetricVirtualPool:
    """Default AVMP v2 dynamic variant: rebalance_enabled=True, all other
    knobs at the RFC 0002 section 4.2 defaults."""

    return AsymmetricVirtualPool(
        model_spec=_hybrid_spec_from_workload(spec),
        total_bytes=_DEFAULT_TOTAL_BYTES,
        device=device,
        rebalance_enabled=True,
    )


register_allocator("padded_unified", _padded_unified_factory)
register_allocator("fixed_dual", _fixed_dual_factory)
register_allocator("avmp_static", _avmp_static_factory)
register_allocator("avmp_dynamic", _avmp_dynamic_factory)


__all__ = [
    "JAMBA_MINI_ATTN",
    "JAMBA_MINI_SSM",
    "MAMBA2_1B3_REF",
    "PRESETS",
    "REGISTRY",
    "SCHEMA_VERSION",
    "AllocatorFactory",
    "AllocatorMetrics",
    "AsymmetricVirtualPool",
    "AttentionLayerProfile",
    "BenchmarkRun",
    "Environment",
    "FixedDualPool",
    "Hardware",
    "HybridModelSpec",
    "LatencyPercentiles",
    "LayerKind",
    "LayerSpec",
    "MetricsCollector",
    "PaddedUnifiedPool",
    "Request",
    "SSMLayerProfile",
    "WorkloadSpec",
    "compute_percentiles",
    "generate_request_stream",
    "per_sequence_ssm_bytes",
    "per_token_kv_bytes",
    "register_allocator",
    "run_benchmark",
]
