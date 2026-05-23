"""Planner comparison helpers for vLLM-style padding vs Cachepawl AVMP."""

from __future__ import annotations

from hashlib import sha256

from cachepawl.bench.environment import RuntimeEnvironment
from cachepawl.bench.planner_baselines import (
    COMPARISON_BACKENDS,
    PlannerBackend,
    estimate_planner,
)
from cachepawl.bench.result_schema import CacheProbeResult, Metadata
from cachepawl.bench.synthetic_workloads import (
    DEFAULT_ATTENTION_PAGE_TOKENS,
    DEFAULT_TIMESTAMP,
    SYNTHETIC_WORKLOADS,
    SyntheticWorkload,
    SyntheticWorkloadName,
    generate_synthetic_workload,
)
from cachepawl.models.spec import JAMBA_1_5_MINI_REF, HybridModelSpec


def compare_planners(
    *,
    workloads: tuple[SyntheticWorkloadName, ...] = SYNTHETIC_WORKLOADS,
    backends: tuple[PlannerBackend, ...] = COMPARISON_BACKENDS,
    seed: int,
    num_requests: int,
    environment: RuntimeEnvironment,
    model: HybridModelSpec = JAMBA_1_5_MINI_REF,
    timestamp: str = DEFAULT_TIMESTAMP,
    attention_page_tokens: int = DEFAULT_ATTENTION_PAGE_TOKENS,
    measure_runtime: bool = False,
) -> tuple[CacheProbeResult, ...]:
    """Run planner comparison records for shared deterministic workloads."""

    records: list[CacheProbeResult] = []
    for workload_name in workloads:
        workload = generate_synthetic_workload(
            workload_name,
            seed=seed,
            num_requests=num_requests,
        )
        for backend in backends:
            records.append(
                result_from_estimate(
                    backend=backend,
                    workload=workload,
                    environment=environment,
                    model=model,
                    timestamp=timestamp,
                    attention_page_tokens=attention_page_tokens,
                    measure_runtime=measure_runtime,
                )
            )
    return tuple(records)


def result_from_estimate(
    *,
    backend: PlannerBackend,
    workload: SyntheticWorkload,
    environment: RuntimeEnvironment,
    model: HybridModelSpec,
    timestamp: str,
    attention_page_tokens: int,
    measure_runtime: bool,
) -> CacheProbeResult:
    estimate = estimate_planner(
        backend,
        workload,
        model=model,
        gpu_total_bytes=environment.gpu.total_memory_bytes,
        attention_page_tokens=attention_page_tokens,
        measure_runtime=measure_runtime,
    )
    metadata: Metadata = {
        **environment.metadata,
        **estimate.metadata,
        "num_requests": workload.num_requests,
        "seed": workload.seed,
    }
    return CacheProbeResult(
        run_id=_run_id(
            timestamp=timestamp,
            backend=estimate.backend,
            workload=workload.name,
            model=model.name,
            seed=workload.seed,
            num_requests=workload.num_requests,
        ),
        timestamp=timestamp,
        backend=estimate.backend,
        workload=workload.name,
        model=model.name,
        gpu=environment.gpu,
        estimated_bytes=estimate.estimated_bytes,
        reserved_bytes=estimate.reserved_bytes,
        useful_bytes=estimate.useful_bytes,
        overestimation_ratio=estimate.overestimation_ratio,
        wasted_fraction=estimate.wasted_fraction,
        virtual_oom=estimate.virtual_oom,
        planner_runtime_us=estimate.planner_runtime_us,
        metadata=metadata,
    )


def render_markdown_summary(records: tuple[CacheProbeResult, ...]) -> str:
    """Render a compact Markdown comparison summary."""

    header = (
        "| backend | workload | useful_bytes | estimated_bytes | overestimation_ratio | "
        "wasted_fraction | virtual_oom | planner_runtime_us |"
    )
    lines = [
        header,
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record.backend} | {record.workload} | {record.useful_bytes} | "
            f"{record.estimated_bytes} | {record.overestimation_ratio:.6f} | "
            f"{record.wasted_fraction:.6f} | "
            f"{str(record.virtual_oom).lower()} | {record.planner_runtime_us:.3f} |"
        )
    return "\n".join(lines) + "\n"


def render_csv_summary(records: tuple[CacheProbeResult, ...]) -> str:
    """Render a compact CSV comparison summary."""

    lines = [
        "backend,workload,useful_bytes,estimated_bytes,overestimation_ratio,"
        "wasted_fraction,virtual_oom,planner_runtime_us"
    ]
    for record in records:
        lines.append(
            f"{record.backend},{record.workload},{record.useful_bytes},"
            f"{record.estimated_bytes},{record.overestimation_ratio:.6f},"
            f"{record.wasted_fraction:.6f},"
            f"{str(record.virtual_oom).lower()},{record.planner_runtime_us:.3f}"
        )
    return "\n".join(lines) + "\n"


def render_jsonl(records: tuple[CacheProbeResult, ...]) -> str:
    return "\n".join(record.to_json_line() for record in records) + ("\n" if records else "")


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
