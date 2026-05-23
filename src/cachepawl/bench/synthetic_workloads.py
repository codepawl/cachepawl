"""Deterministic planner-only synthetic cache workloads."""

from __future__ import annotations

import random
from dataclasses import dataclass
from hashlib import sha256
from typing import Final, Literal, TypeAlias

from cachepawl.bench.environment import RuntimeEnvironment
from cachepawl.bench.planner_baselines import (
    PlannerBackend,
    estimate_planner,
    overestimation_ratio,
    wasted_fraction,
)
from cachepawl.bench.result_schema import CacheProbeResult, Metadata
from cachepawl.models.spec import JAMBA_1_5_MINI_REF, HybridModelSpec

SyntheticWorkloadName: TypeAlias = Literal["short-heavy", "long-heavy", "mixed"]
ProbeBackend: TypeAlias = Literal["padded-unified", "avmp-static", "fixed-dual"]
SYNTHETIC_WORKLOADS: Final[tuple[SyntheticWorkloadName, ...]] = (
    "short-heavy",
    "long-heavy",
    "mixed",
)
PLANNER_BACKENDS: Final[tuple[ProbeBackend, ...]] = (
    "padded-unified",
    "avmp-static",
    "fixed-dual",
)
DEFAULT_TIMESTAMP: Final[str] = "1970-01-01T00:00:00Z"
DEFAULT_ATTENTION_PAGE_TOKENS: Final[int] = 16
DEFAULT_FIXED_DUAL_MAMBA_RATIO: Final[float] = 0.5


@dataclass(frozen=True, slots=True)
class SyntheticRequest:
    """One deterministic synthetic cache request."""

    request_id: int
    prompt_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class SyntheticWorkload:
    """A named deterministic synthetic workload."""

    name: SyntheticWorkloadName
    seed: int
    requests: tuple[SyntheticRequest, ...]

    @property
    def num_requests(self) -> int:
        return len(self.requests)

    @property
    def total_tokens(self) -> int:
        return sum(request.total_tokens for request in self.requests)


def generate_synthetic_workload(
    name: SyntheticWorkloadName,
    *,
    seed: int,
    num_requests: int,
) -> SyntheticWorkload:
    """Generate a deterministic synthetic workload by name."""

    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")

    rng = random.Random(seed)
    requests: list[SyntheticRequest] = []
    for request_id in range(num_requests):
        if name == "short-heavy":
            prompt_tokens = rng.randint(64, 1024)
            output_tokens = rng.randint(32, 256)
        elif name == "long-heavy":
            prompt_tokens = rng.randint(4096, 32768)
            output_tokens = rng.randint(128, 1024)
        elif name == "mixed":
            if rng.random() < 0.7:
                prompt_tokens = rng.randint(128, 2048)
                output_tokens = rng.randint(32, 384)
            else:
                prompt_tokens = rng.randint(4096, 24576)
                output_tokens = rng.randint(128, 1024)
        else:
            raise ValueError(f"unknown synthetic workload {name!r}")
        requests.append(
            SyntheticRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        )
    return SyntheticWorkload(name=name, seed=seed, requests=tuple(requests))


def build_probe_result(
    *,
    backend: ProbeBackend,
    workload: SyntheticWorkload,
    environment: RuntimeEnvironment,
    model: HybridModelSpec = JAMBA_1_5_MINI_REF,
    timestamp: str = DEFAULT_TIMESTAMP,
    attention_page_tokens: int = DEFAULT_ATTENTION_PAGE_TOKENS,
    fixed_dual_mamba_ratio: float = DEFAULT_FIXED_DUAL_MAMBA_RATIO,
    measure_runtime: bool = False,
) -> CacheProbeResult:
    """Build one deterministic planner benchmark record."""

    if attention_page_tokens <= 0:
        raise ValueError("attention_page_tokens must be positive")
    if not 0.0 < fixed_dual_mamba_ratio < 1.0:
        raise ValueError("fixed_dual_mamba_ratio must be in (0, 1)")

    comparison_backend: PlannerBackend
    if backend == "padded-unified":
        comparison_backend = "vllm-style-padded"
    elif backend in {"avmp-static", "fixed-dual"}:
        comparison_backend = "cachepawl-avmp"
    else:
        raise ValueError(f"unknown planner backend {backend!r}")

    estimate = estimate_planner(
        comparison_backend,
        workload,
        model=model,
        gpu_total_bytes=environment.gpu.total_memory_bytes,
        attention_page_tokens=attention_page_tokens,
        measure_runtime=measure_runtime,
    )
    reserved_bytes = estimate.reserved_bytes
    if backend == "fixed-dual":
        reserved_bytes = int(
            reserved_bytes / min(fixed_dual_mamba_ratio, 1.0 - fixed_dual_mamba_ratio)
        )
    metadata: Metadata = {
        **environment.metadata,
        **estimate.metadata,
        "num_requests": workload.num_requests,
        "seed": workload.seed,
        "attention_page_tokens": attention_page_tokens,
        "fixed_dual_mamba_ratio": fixed_dual_mamba_ratio,
    }
    return CacheProbeResult(
        run_id=_run_id(
            timestamp=timestamp,
            backend=backend,
            workload=workload.name,
            model=model.name,
            seed=workload.seed,
            num_requests=workload.num_requests,
        ),
        timestamp=timestamp,
        backend=backend,
        workload=workload.name,
        model=model.name,
        gpu=environment.gpu,
        estimated_bytes=reserved_bytes,
        reserved_bytes=reserved_bytes,
        useful_bytes=estimate.useful_bytes,
        overestimation_ratio=overestimation_ratio(
            estimated_bytes=reserved_bytes,
            useful_bytes=estimate.useful_bytes,
        ),
        wasted_fraction=wasted_fraction(
            estimated_bytes=reserved_bytes,
            useful_bytes=estimate.useful_bytes,
        ),
        virtual_oom=(
            environment.gpu.total_memory_bytes is not None
            and reserved_bytes > environment.gpu.total_memory_bytes
        ),
        planner_runtime_us=estimate.planner_runtime_us,
        metadata=metadata,
    )


def _run_id(
    *,
    timestamp: str,
    backend: str,
    workload: str,
    model: str,
    seed: int,
    num_requests: int,
) -> str:
    material = f"{timestamp}|{backend}|{workload}|{model}|{seed}|{num_requests}"
    return sha256(material.encode("utf-8")).hexdigest()[:16]
