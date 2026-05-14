"""Shared fixtures for the comparison sweep unit tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from cachepawl.benchmarks import (
    JAMBA_MINI_ATTN,
    JAMBA_MINI_SSM,
    AllocatorMetrics,
    BenchmarkRun,
    Environment,
    Hardware,
    WorkloadSpec,
)
from cachepawl.benchmarks.compare.sweep import (
    AllocatorVariant,
    SweepConfig,
    SweepMetadata,
    SweepResult,
)
from cachepawl.quant.dtypes import DType


def make_run(
    *,
    allocator_label: str,
    workload_name: str,
    seed: int,
    peak_reserved_bytes: int,
    final_fragmentation: float,
    oom_count: int,
    allocator_specific_stats: Mapping[str, float],
    allocate_latency_ns: list[int] | None = None,
    fragmentation_samples: list[float] | None = None,
    active_requests_samples: list[int] | None = None,
) -> BenchmarkRun:
    """Build a BenchmarkRun with deliberately chosen metric values.

    By default, ``fragmentation_samples`` is a single tick with the
    provided ``final_fragmentation`` value, paired against a single
    ``active_requests_samples`` entry of ``1`` so the during-load
    filter keeps it. Callers that want to exercise the teardown filter
    can pass their own parallel lists explicitly.
    """

    if fragmentation_samples is None:
        fragmentation_samples = [final_fragmentation]
    if active_requests_samples is None:
        active_requests_samples = [1] * len(fragmentation_samples)
    spec = WorkloadSpec(
        name=workload_name,
        num_requests=16,
        attention_layers=4,
        ssm_layers=28,
        attention_profile=JAMBA_MINI_ATTN,
        ssm_profile=JAMBA_MINI_SSM,
        dtype=DType.BF16,
        seed=seed,
    )
    metrics = AllocatorMetrics(
        peak_reserved_bytes=peak_reserved_bytes,
        peak_allocated_bytes=0,
        fragmentation_samples=list(fragmentation_samples),
        allocate_latency_ns=list(allocate_latency_ns or [1000, 2000, 3000]),
        free_latency_ns=[500, 700, 900],
        oom_count=oom_count,
        preemption_count=0,
        active_requests_samples=list(active_requests_samples),
        allocator_specific_stats=dict(allocator_specific_stats),
    )
    return BenchmarkRun(
        spec=spec,
        allocator_name=allocator_label,
        hardware=Hardware(device="cpu", gpu_name=None, vram_total_bytes=None, cuda_capability=None),
        environment=Environment(
            torch_version="0.0",
            numpy_version="0.0",
            cachepawl_version="0.0",
            cuda_version=None,
            python_version="3.10",
        ),
        started_at="2026-05-14T00:00:00Z",
        finished_at="2026-05-14T00:00:01Z",
        metrics=metrics,
    )


def make_sweep_result(
    *,
    config: SweepConfig,
    runs: list[BenchmarkRun],
    cell_stems: list[str],
    failures: list[object] | None = None,
) -> SweepResult:
    """Wrap hand-built runs in a SweepResult so aggregate_runs can consume it."""

    if len(runs) != len(cell_stems):
        raise ValueError("runs and cell_stems must have matching length")
    metadata = SweepMetadata(
        git_sha="0" * 40,
        torch_version="0.0",
        numpy_version="0.0",
        python_version="3.10",
        cachepawl_version="0.0",
        cuda_version=None,
        gpu_name=None,
        device=config.device,
        hardware_label="cpu (test)",
        sweep_started_at="2026-05-14T00:00:00Z",
        sweep_finished_at="2026-05-14T00:00:01Z",
        total_wall_seconds=1.0,
        n_cells_planned=len(runs),
        n_cells_succeeded=len(runs),
        n_cells_failed=0,
    )
    return SweepResult(
        config=config,
        runs=list(runs),
        failures=list(failures or []),  # type: ignore[arg-type]
        metadata=metadata,
        cell_stems={i: stem for i, stem in enumerate(cell_stems)},
    )


@pytest.fixture
def default_config(tmp_path: Path) -> SweepConfig:
    """A minimal SweepConfig pointing at tmp_path; tests rarely run it."""

    return SweepConfig(
        variants=(
            AllocatorVariant(label="padded_unified", allocator_name="padded_unified", kwargs=()),
            AllocatorVariant(
                label="fixed_dual_mr05",
                allocator_name="fixed_dual",
                kwargs=(("mamba_ratio", 0.5),),
            ),
        ),
        workload_names=("uniform_short",),
        model_spec_names=("jamba_1_5_mini",),
        total_bytes_options=(1 * 1024**3,),
        device="cpu",
        output_dir=tmp_path,
        seed_replicates=3,
        smoke_num_requests=None,
    )
