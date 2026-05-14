"""Aggregation math: mean/std/percentiles match hand-computed values."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cachepawl.benchmarks import BenchmarkRun
from cachepawl.benchmarks.compare import (
    AggregatedRow,
    SweepConfig,
    aggregate_runs,
    compute_relative_improvement,
)
from cachepawl.benchmarks.compare.sweep import AllocatorVariant
from tests.unit.benchmarks.compare.conftest import make_run, make_sweep_result


def _build_replicates(
    *,
    config: SweepConfig,
    allocator_label: str,
    workload_name: str,
    model_spec_name: str,
    total_bytes: int,
    peak_values: list[int],
    final_frags: list[float],
    oom_values: list[int],
    allocator_specifics: list[dict[str, float]],
) -> tuple[list[BenchmarkRun], list[str]]:
    runs: list[BenchmarkRun] = []
    stems: list[str] = []
    for idx, (peak, frag, oom, specific) in enumerate(
        zip(peak_values, final_frags, oom_values, allocator_specifics, strict=True)
    ):
        runs.append(
            make_run(
                allocator_label=allocator_label,
                workload_name=workload_name,
                seed=idx + 1,
                peak_reserved_bytes=peak,
                final_fragmentation=frag,
                oom_count=oom,
                allocator_specific_stats=specific,
            )
        )
        stems.append(f"{model_spec_name}__tb{total_bytes // 1024**3}gib__seed{idx + 1}")
    return runs, stems


def test_aggregate_three_replicates_mean_and_std() -> None:
    """peak_reserved_bytes mean/std/min/max match numpy.mean/std (ddof=0)."""

    config = SweepConfig(
        variants=(AllocatorVariant("padded_unified", "padded_unified", ()),),
        workload_names=("uniform_short",),
        model_spec_names=("jamba_1_5_mini",),
        total_bytes_options=(1 * 1024**3,),
        device="cpu",
        output_dir=Path("/tmp/probe"),
        seed_replicates=3,
    )
    runs, stems = _build_replicates(
        config=config,
        allocator_label="padded_unified",
        workload_name="uniform_short",
        model_spec_name="jamba_1_5_mini",
        total_bytes=1 * 1024**3,
        peak_values=[100, 200, 300],
        final_frags=[0.10, 0.20, 0.30],
        oom_values=[0, 0, 0],
        allocator_specifics=[{"padding_waste_bytes": v} for v in (10.0, 20.0, 30.0)],
    )
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)
    assert len(aggregated.rows) == 1
    row = aggregated.rows[0]
    assert row.n_replicates == 3
    assert row.peak_reserved_bytes_mean == pytest.approx(200.0)
    expected_std = math.sqrt(((100 - 200) ** 2 + 0 + (300 - 200) ** 2) / 3.0)
    assert row.peak_reserved_bytes_std == pytest.approx(expected_std)
    assert row.peak_reserved_bytes_min == 100
    assert row.peak_reserved_bytes_max == 300
    assert row.fragmentation_final_mean == pytest.approx(0.20)
    assert row.oom_count_mean == pytest.approx(0.0)


def test_aggregate_allocator_specific_median_and_iqr() -> None:
    """Median is the middle value, IQR is the 75-25 percentile gap."""

    config = SweepConfig(
        variants=(AllocatorVariant("padded_unified", "padded_unified", ()),),
        workload_names=("uniform_short",),
        model_spec_names=("jamba_1_5_mini",),
        total_bytes_options=(1 * 1024**3,),
        device="cpu",
        output_dir=Path("/tmp/probe"),
        seed_replicates=3,
    )
    runs, stems = _build_replicates(
        config=config,
        allocator_label="padded_unified",
        workload_name="uniform_short",
        model_spec_name="jamba_1_5_mini",
        total_bytes=1 * 1024**3,
        peak_values=[100, 100, 100],
        final_frags=[0.10, 0.10, 0.10],
        oom_values=[0, 0, 0],
        allocator_specifics=[{"padding_waste_bytes": v} for v in (10.0, 20.0, 30.0)],
    )
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)
    medians = dict(aggregated.rows[0].allocator_specific_median)
    iqrs = dict(aggregated.rows[0].allocator_specific_iqr)
    assert medians["padding_waste_bytes"] == pytest.approx(20.0)
    assert iqrs["padding_waste_bytes"] == pytest.approx(10.0)


def test_aggregate_groups_replicates_correctly() -> None:
    """Two variants with three replicates each produce two AggregatedRows."""

    config = SweepConfig(
        variants=(
            AllocatorVariant("padded_unified", "padded_unified", ()),
            AllocatorVariant("fixed_dual_mr05", "fixed_dual", (("mamba_ratio", 0.5),)),
        ),
        workload_names=("uniform_short",),
        model_spec_names=("jamba_1_5_mini",),
        total_bytes_options=(1 * 1024**3,),
        device="cpu",
        output_dir=Path("/tmp/probe"),
        seed_replicates=3,
    )
    runs_padded, stems_padded = _build_replicates(
        config=config,
        allocator_label="padded_unified",
        workload_name="uniform_short",
        model_spec_name="jamba_1_5_mini",
        total_bytes=1 * 1024**3,
        peak_values=[100, 100, 100],
        final_frags=[0.10, 0.10, 0.10],
        oom_values=[0, 0, 0],
        allocator_specifics=[{"padding_waste_bytes": 5.0}] * 3,
    )
    runs_dual, stems_dual = _build_replicates(
        config=config,
        allocator_label="fixed_dual_mr05",
        workload_name="uniform_short",
        model_spec_name="jamba_1_5_mini",
        total_bytes=1 * 1024**3,
        peak_values=[200, 200, 200],
        final_frags=[0.05, 0.05, 0.05],
        oom_values=[1, 1, 1],
        allocator_specifics=[{"pool_underused_bytes_kv": 3.0, "pool_underused_bytes_ssm": 7.0}] * 3,
    )
    result = make_sweep_result(
        config=config,
        runs=runs_padded + runs_dual,
        cell_stems=stems_padded + stems_dual,
    )
    aggregated = aggregate_runs(result)
    assert len(aggregated.rows) == 2
    labels = {row.variant_label for row in aggregated.rows}
    assert labels == {"padded_unified", "fixed_dual_mr05"}


def test_compute_relative_improvement_lower_is_better() -> None:
    """target with 0.02 frag vs baseline 0.05 -> 60% better."""

    def _row(label: str, frag: float, peak: float, oom: float) -> AggregatedRow:
        return AggregatedRow(
            variant_label=label,
            allocator_name="x",
            workload_name="uniform_short",
            model_spec_name="jamba_1_5_mini",
            total_bytes=1 * 1024**3,
            n_replicates=1,
            peak_reserved_bytes_mean=peak,
            peak_reserved_bytes_std=0.0,
            peak_reserved_bytes_min=int(peak),
            peak_reserved_bytes_max=int(peak),
            fragmentation_final_mean=frag,
            fragmentation_final_std=0.0,
            fragmentation_final_min=frag,
            fragmentation_final_max=frag,
            oom_count_mean=oom,
            oom_count_std=0.0,
            allocator_specific_median=(),
            allocator_specific_iqr=(),
            allocate_p50_ns_median=0,
            allocate_p95_ns_median=0,
            allocate_p99_ns_median=0,
        )

    baseline = [_row("padded_unified", 0.05, 100.0, 10.0)]
    target = [_row("fixed_dual_mr05", 0.02, 50.0, 5.0)]
    deltas = compute_relative_improvement(baseline, target)
    key = ("uniform_short", "jamba_1_5_mini", "fixed_dual_mr05", 1 * 1024**3)
    assert deltas[key]["fragmentation_final_mean"] == pytest.approx(60.0)
    assert deltas[key]["peak_reserved_bytes_mean"] == pytest.approx(50.0)
    assert deltas[key]["oom_count_mean"] == pytest.approx(50.0)


def test_compute_relative_improvement_rejects_duplicate_baseline_cells() -> None:
    """Two baseline rows at the same cell key is a programmer error."""

    def _row(label: str) -> AggregatedRow:
        return AggregatedRow(
            variant_label=label,
            allocator_name="x",
            workload_name="uniform_short",
            model_spec_name="jamba_1_5_mini",
            total_bytes=1 * 1024**3,
            n_replicates=1,
            peak_reserved_bytes_mean=0.0,
            peak_reserved_bytes_std=0.0,
            peak_reserved_bytes_min=0,
            peak_reserved_bytes_max=0,
            fragmentation_final_mean=0.0,
            fragmentation_final_std=0.0,
            fragmentation_final_min=0.0,
            fragmentation_final_max=0.0,
            oom_count_mean=0.0,
            oom_count_std=0.0,
            allocator_specific_median=(),
            allocator_specific_iqr=(),
            allocate_p50_ns_median=0,
            allocate_p95_ns_median=0,
            allocate_p99_ns_median=0,
        )

    with pytest.raises(ValueError, match="duplicate cell"):
        compute_relative_improvement([_row("a"), _row("a")], [_row("b")])
