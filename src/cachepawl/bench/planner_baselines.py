"""Planner-only cache memory estimates for comparison probes."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

from cachepawl.models.spec import JAMBA_1_5_MINI_REF, HybridModelSpec
from cachepawl.quant.dtypes import bytes_per_element

from .result_schema import Metadata

if TYPE_CHECKING:
    from .synthetic_workloads import SyntheticWorkload

PlannerBackend: TypeAlias = Literal["vllm-style-padded", "cachepawl-avmp"]
COMPARISON_BACKENDS: tuple[PlannerBackend, ...] = ("vllm-style-padded", "cachepawl-avmp")
_ALIGNMENT_BYTES: int = 128
_DEFAULT_ATTENTION_PAGE_TOKENS: int = 16


@dataclass(frozen=True, slots=True)
class PlannerEstimate:
    """One planner memory estimate before schema serialization."""

    backend: PlannerBackend
    estimated_bytes: int
    reserved_bytes: int
    useful_bytes: int
    overestimation_ratio: float
    wasted_fraction: float
    virtual_oom: bool
    planner_runtime_us: float
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.estimated_bytes < 0:
            raise ValueError("estimated_bytes must be non-negative")
        if self.reserved_bytes < 0:
            raise ValueError("reserved_bytes must be non-negative")
        if self.useful_bytes < 0:
            raise ValueError("useful_bytes must be non-negative")
        if self.useful_bytes == 0 and self.estimated_bytes > 0:
            raise ValueError("useful_bytes must be positive when estimated_bytes is positive")
        if self.overestimation_ratio < 0.0:
            raise ValueError("overestimation_ratio must be non-negative")
        if not 0.0 <= self.wasted_fraction <= 1.0:
            raise ValueError("wasted_fraction must be in [0.0, 1.0]")
        if self.planner_runtime_us < 0.0:
            raise ValueError("planner_runtime_us must be non-negative")


def estimate_vllm_style_padded(
    workload: SyntheticWorkload,
    *,
    model: HybridModelSpec = JAMBA_1_5_MINI_REF,
    gpu_total_bytes: int | None = None,
    attention_page_tokens: int = _DEFAULT_ATTENTION_PAGE_TOKENS,
    measure_runtime: bool = False,
) -> PlannerEstimate:
    """Estimate a vLLM-style uniform page-size modeling baseline.

    This is a modeling baseline for heterogeneous KV/SSM padding. It is
    intentionally not an exact claim about current vLLM internals.
    """

    start_ns = time.perf_counter_ns() if measure_runtime else 0
    useful_bytes = useful_cache_bytes(model, workload)
    attention_layers = attention_layer_count(model)
    ssm_layers = len(model.layers) - attention_layers
    kv_page_bytes = align_up(math.ceil(kv_bytes_per_token(model) * attention_page_tokens))
    ssm_block_bytes = align_up(math.ceil(ssm_bytes_per_sequence(model)))
    unified_page_bytes = max(kv_page_bytes, ssm_block_bytes)
    kv_pages = sum(
        math.ceil(request.total_tokens / attention_page_tokens) * attention_layers
        for request in workload.requests
    )
    ssm_blocks = len(workload.requests) * ssm_layers
    reserved_bytes = (kv_pages + ssm_blocks) * unified_page_bytes
    runtime_us = (time.perf_counter_ns() - start_ns) / 1000.0 if measure_runtime else 0.0
    return PlannerEstimate(
        backend="vllm-style-padded",
        estimated_bytes=reserved_bytes,
        reserved_bytes=reserved_bytes,
        useful_bytes=useful_bytes,
        overestimation_ratio=overestimation_ratio(
            estimated_bytes=reserved_bytes,
            useful_bytes=useful_bytes,
        ),
        wasted_fraction=wasted_fraction(
            estimated_bytes=reserved_bytes,
            useful_bytes=useful_bytes,
        ),
        virtual_oom=(gpu_total_bytes is not None and reserved_bytes > gpu_total_bytes),
        planner_runtime_us=runtime_us,
        metadata={
            "planner_model": "uniform-page-padding",
            "attention_page_tokens": attention_page_tokens,
            "kv_page_bytes": kv_page_bytes,
            "ssm_block_bytes": ssm_block_bytes,
            "unified_page_bytes": unified_page_bytes,
            "kv_pages": kv_pages,
            "ssm_blocks": ssm_blocks,
        },
    )


def estimate_avmp_static(
    workload: SyntheticWorkload,
    *,
    model: HybridModelSpec = JAMBA_1_5_MINI_REF,
    gpu_total_bytes: int | None = None,
    attention_page_tokens: int = _DEFAULT_ATTENTION_PAGE_TOKENS,
    measure_runtime: bool = False,
) -> PlannerEstimate:
    """Estimate Cachepawl AVMP native KV-page and SSM-block planning."""

    start_ns = time.perf_counter_ns() if measure_runtime else 0
    useful_bytes = useful_cache_bytes(model, workload)
    attention_layers = attention_layer_count(model)
    ssm_layers = len(model.layers) - attention_layers
    kv_page_bytes = align_up(math.ceil(kv_bytes_per_token(model) * attention_page_tokens))
    ssm_block_bytes = align_up(math.ceil(ssm_bytes_per_sequence(model)))
    kv_pages = sum(
        math.ceil(request.total_tokens / attention_page_tokens) * attention_layers
        for request in workload.requests
    )
    ssm_blocks = len(workload.requests) * ssm_layers
    reserved_bytes = (kv_pages * kv_page_bytes) + (ssm_blocks * ssm_block_bytes)
    runtime_us = (time.perf_counter_ns() - start_ns) / 1000.0 if measure_runtime else 0.0
    return PlannerEstimate(
        backend="cachepawl-avmp",
        estimated_bytes=reserved_bytes,
        reserved_bytes=reserved_bytes,
        useful_bytes=useful_bytes,
        overestimation_ratio=overestimation_ratio(
            estimated_bytes=reserved_bytes,
            useful_bytes=useful_bytes,
        ),
        wasted_fraction=wasted_fraction(
            estimated_bytes=reserved_bytes,
            useful_bytes=useful_bytes,
        ),
        virtual_oom=(gpu_total_bytes is not None and reserved_bytes > gpu_total_bytes),
        planner_runtime_us=runtime_us,
        metadata={
            "planner_model": "native-kv-page-ssm-block",
            "attention_page_tokens": attention_page_tokens,
            "kv_page_bytes": kv_page_bytes,
            "ssm_block_bytes": ssm_block_bytes,
            "kv_pages": kv_pages,
            "ssm_blocks": ssm_blocks,
        },
    )


def estimate_planner(
    backend: PlannerBackend,
    workload: SyntheticWorkload,
    *,
    model: HybridModelSpec = JAMBA_1_5_MINI_REF,
    gpu_total_bytes: int | None = None,
    attention_page_tokens: int = _DEFAULT_ATTENTION_PAGE_TOKENS,
    measure_runtime: bool = False,
) -> PlannerEstimate:
    """Dispatch one named planner estimate."""

    if backend == "vllm-style-padded":
        return estimate_vllm_style_padded(
            workload,
            model=model,
            gpu_total_bytes=gpu_total_bytes,
            attention_page_tokens=attention_page_tokens,
            measure_runtime=measure_runtime,
        )
    if backend == "cachepawl-avmp":
        return estimate_avmp_static(
            workload,
            model=model,
            gpu_total_bytes=gpu_total_bytes,
            attention_page_tokens=attention_page_tokens,
            measure_runtime=measure_runtime,
        )
    raise ValueError(f"unknown planner backend {backend!r}")


def useful_cache_bytes(model: HybridModelSpec, workload: SyntheticWorkload) -> int:
    """Logical useful cache bytes before backend-specific reservation."""

    attention_layers = attention_layer_count(model)
    ssm_layers = len(model.layers) - attention_layers
    kv_per_token = kv_bytes_per_token(model)
    ssm_per_sequence = ssm_bytes_per_sequence(model)
    kv_bytes = sum(
        math.ceil(request.total_tokens * kv_per_token) * attention_layers
        for request in workload.requests
    )
    ssm_bytes = sum(math.ceil(ssm_per_sequence) * ssm_layers for _ in workload.requests)
    return kv_bytes + ssm_bytes


def kv_bytes_per_token(model: HybridModelSpec) -> float:
    profile = model.attention_profile
    return 2.0 * profile.num_kv_heads * profile.head_dim * bytes_per_element(model.dtype)


def ssm_bytes_per_sequence(model: HybridModelSpec) -> float:
    profile = model.ssm_profile
    return float(profile.d_inner) * float(profile.d_state) * bytes_per_element(model.dtype)


def attention_layer_count(model: HybridModelSpec) -> int:
    return sum(1 for layer in model.layers if layer.kind.value.endswith("attention"))


def align_up(value: int, alignment: int = _ALIGNMENT_BYTES) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def overestimation_ratio(*, estimated_bytes: int, useful_bytes: int) -> float:
    if useful_bytes == 0:
        if estimated_bytes == 0:
            return 0.0
        raise ValueError("useful_bytes must be positive when estimated_bytes is positive")
    return estimated_bytes / useful_bytes


def wasted_fraction(*, estimated_bytes: int, useful_bytes: int) -> float:
    if estimated_bytes == 0:
        return 0.0
    return min(1.0, max(0.0, (estimated_bytes - useful_bytes) / estimated_bytes))
