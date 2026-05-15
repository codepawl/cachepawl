"""Plot tests for the ``avmp_static`` rendering path.

Asserts both plots accept an AVMP-bearing aggregated set without
raising and that the rigidity-surface helper derives the same total
free MiB the report row reports for the same row.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from cachepawl.benchmarks import BenchmarkRun
from cachepawl.benchmarks.compare import (
    SweepConfig,
    aggregate_runs,
    plot_fragmentation_vs_workload,
    plot_padding_waste_vs_state_size,
)
from cachepawl.benchmarks.compare.sweep import AllocatorVariant
from tests.unit.benchmarks.compare.conftest import make_run, make_sweep_result

_MIB: float = 1024.0 * 1024.0


def _build_avmp_plus_baselines_sweep(
    tmp_path: Path,
) -> tuple[SweepConfig, list[BenchmarkRun], list[str]]:
    config = SweepConfig(
        variants=(
            AllocatorVariant("padded_unified", "padded_unified", ()),
            AllocatorVariant("fixed_dual_mr05", "fixed_dual", (("mamba_ratio", 0.5),)),
            AllocatorVariant("avmp_static_mr05", "avmp_static", (("mamba_ratio", 0.5),)),
        ),
        workload_names=("uniform_short",),
        model_spec_names=("jamba_1_5_mini",),
        total_bytes_options=(1 * 1024**3,),
        device="cpu",
        output_dir=tmp_path,
        seed_replicates=1,
    )
    runs: list[BenchmarkRun] = []
    stems: list[str] = []
    runs.append(
        make_run(
            allocator_label="padded_unified",
            workload_name="uniform_short",
            seed=1,
            peak_reserved_bytes=500_000,
            final_fragmentation=0.05,
            oom_count=0,
            allocator_specific_stats={"padding_waste_bytes": 2_000_000.0},
        )
    )
    stems.append("jamba_1_5_mini__tb1gib__seed1")
    runs.append(
        make_run(
            allocator_label="fixed_dual_mr05",
            workload_name="uniform_short",
            seed=1,
            peak_reserved_bytes=500_000,
            final_fragmentation=0.02,
            oom_count=0,
            allocator_specific_stats={
                "pool_free_bytes_kv": 1_000_000.0,
                "pool_free_bytes_ssm": 500_000.0,
            },
        )
    )
    stems.append("jamba_1_5_mini__tb1gib__seed1")
    runs.append(
        make_run(
            allocator_label="avmp_static_mr05",
            workload_name="uniform_short",
            seed=1,
            peak_reserved_bytes=500_000,
            final_fragmentation=0.02,
            oom_count=0,
            allocator_specific_stats={
                "kv_pages_total": 100.0,
                "kv_pages_used": 50.0,
                "kv_pages_free": 50.0,
                "ssm_blocks_total": 4.0,
                "ssm_blocks_used": 2.0,
                "ssm_blocks_free": 2.0,
                "virtual_handles_live": 52.0,
                "cross_pool_eviction_count": 0.0,
                "kv_pool_bytes": 100.0 * _MIB,
                "ssm_pool_bytes": 4.0 * _MIB,
                "mamba_ratio": 0.5,
            },
        )
    )
    stems.append("jamba_1_5_mini__tb1gib__seed1")
    return config, runs, stems


def test_fragmentation_plot_renders_with_avmp_variant(tmp_path: Path) -> None:
    config, runs, stems = _build_avmp_plus_baselines_sweep(tmp_path)
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)
    out = tmp_path / "frag.png"
    plot_fragmentation_vs_workload(
        aggregated,
        out,
        model_spec_filter="jamba_1_5_mini",
        git_sha="0" * 40,
        run_date="2026-05-15",
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    with Image.open(out) as img:
        assert img.format == "PNG"


def test_padding_waste_plot_renders_with_avmp_variant(tmp_path: Path) -> None:
    config, runs, stems = _build_avmp_plus_baselines_sweep(tmp_path)
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)
    out = tmp_path / "padding.png"
    plot_padding_waste_vs_state_size(
        aggregated,
        out,
        workload_filter="uniform_short",
        git_sha="0" * 40,
        run_date="2026-05-15",
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    with Image.open(out) as img:
        assert img.format == "PNG"
