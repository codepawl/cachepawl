"""Plot generation: PNG file exists and opens via PIL."""

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
from cachepawl.benchmarks.compare.plots import (
    plot_effective_batch_size_vs_workload,
    plot_goodput_comparison,
)
from cachepawl.benchmarks.compare.sweep import AllocatorVariant
from tests.unit.benchmarks.compare.conftest import make_run, make_sweep_result


def _build_two_spec_sweep(
    tmp_path: Path,
) -> tuple[SweepConfig, list[BenchmarkRun], list[str]]:
    config = SweepConfig(
        variants=(
            AllocatorVariant("padded_unified", "padded_unified", ()),
            AllocatorVariant("fixed_dual_mr05", "fixed_dual", (("mamba_ratio", 0.5),)),
        ),
        workload_names=("uniform_short",),
        model_spec_names=("jamba_1_5_mini", "mamba2_1b3"),
        total_bytes_options=(1 * 1024**3,),
        device="cpu",
        output_dir=tmp_path,
        seed_replicates=2,
    )
    runs: list[BenchmarkRun] = []
    stems: list[str] = []
    for variant_label in ("padded_unified", "fixed_dual_mr05"):
        for model_spec in ("jamba_1_5_mini", "mamba2_1b3"):
            for seed in (1, 2):
                specific: dict[str, float]
                if variant_label == "padded_unified":
                    specific = {"padding_waste_bytes": 2_000_000.0}
                else:
                    specific = {
                        "pool_free_bytes_kv": 1_000_000.0,
                        "pool_free_bytes_ssm": 500_000.0,
                    }
                runs.append(
                    make_run(
                        allocator_label=variant_label,
                        workload_name="uniform_short",
                        seed=seed,
                        peak_reserved_bytes=500_000,
                        final_fragmentation=0.05 if variant_label == "padded_unified" else 0.02,
                        oom_count=0,
                        allocator_specific_stats=specific,
                    )
                )
                stems.append(f"{model_spec}__tb1gib__seed{seed}")
    return config, runs, stems


def test_plot_fragmentation_vs_workload_creates_png(tmp_path: Path) -> None:
    config, runs, stems = _build_two_spec_sweep(tmp_path)
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)
    out = tmp_path / "frag.png"
    plot_fragmentation_vs_workload(
        aggregated,
        out,
        model_spec_filter="jamba_1_5_mini",
        git_sha="abc12345" * 5,
        run_date="2026-05-14",
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    with Image.open(out) as img:
        assert img.format == "PNG"
        assert img.size[0] > 200
        assert img.size[1] > 200


def test_plot_padding_waste_vs_state_size_creates_png(tmp_path: Path) -> None:
    config, runs, stems = _build_two_spec_sweep(tmp_path)
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)
    out = tmp_path / "padding.png"
    plot_padding_waste_vs_state_size(
        aggregated,
        out,
        workload_filter="uniform_short",
        git_sha="abc12345" * 5,
        run_date="2026-05-14",
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    with Image.open(out) as img:
        assert img.format == "PNG"


def test_effective_batch_size_plot_emits_png_and_pdf(tmp_path: Path) -> None:
    """The new plot saves both .png and .pdf next to the stem.

    Tier 1 PR B convention: pass output WITHOUT an extension; the
    function writes both formats so the paper build can pick up
    vector output without re-running the sweep.
    """

    config, runs, stems = _build_two_spec_sweep(tmp_path)
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)
    stem = tmp_path / "fig_effective_batch_size_vs_workload"

    plot_effective_batch_size_vs_workload(
        aggregated,
        stem,
        model_spec_filter="jamba_1_5_mini",
        git_sha="abc12345" * 5,
        run_date="2026-05-18",
    )

    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 1024
    assert pdf_path.stat().st_size > 1024


def test_goodput_plot_emits_png_and_pdf(tmp_path: Path) -> None:
    """Mirror coverage for the goodput plot."""

    config, runs, stems = _build_two_spec_sweep(tmp_path)
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)
    stem = tmp_path / "fig_goodput_comparison"

    plot_goodput_comparison(
        aggregated,
        stem,
        model_spec_filter="jamba_1_5_mini",
        git_sha="abc12345" * 5,
        run_date="2026-05-18",
    )

    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    assert png_path.exists()
    assert pdf_path.exists()


def test_new_plots_pdf_has_pdf_magic_header(tmp_path: Path) -> None:
    """The generated PDF starts with the %PDF- magic bytes.

    Matplotlib's Agg backend ships PDF support out of the box, but
    the test pins the format so a CI environment without a working
    PDF backend would be caught here rather than at sweep-commit
    time. PNG presence is covered by the size assertions above; this
    test specifically verifies PDF validity.
    """

    config, runs, stems = _build_two_spec_sweep(tmp_path)
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)
    stem = tmp_path / "fig_eff_batch"

    plot_effective_batch_size_vs_workload(
        aggregated,
        stem,
        model_spec_filter="jamba_1_5_mini",
        git_sha="0" * 40,
        run_date="2026-05-18",
    )

    pdf_path = stem.with_suffix(".pdf")
    head = pdf_path.read_bytes()[:5]
    assert head == b"%PDF-"


def test_plot_uses_largest_total_bytes_when_filter_omitted(tmp_path: Path) -> None:
    """When total_bytes_filter is None the plot picks the largest swept size."""

    config = SweepConfig(
        variants=(AllocatorVariant("padded_unified", "padded_unified", ()),),
        workload_names=("uniform_short",),
        model_spec_names=("jamba_1_5_mini",),
        total_bytes_options=(1 * 1024**3, 4 * 1024**3),
        device="cpu",
        output_dir=tmp_path,
        seed_replicates=1,
    )
    runs: list[BenchmarkRun] = []
    stems: list[str] = []
    for tb_human in ("1gib", "4gib"):
        runs.append(
            make_run(
                allocator_label="padded_unified",
                workload_name="uniform_short",
                seed=1,
                peak_reserved_bytes=100_000,
                final_fragmentation=0.10,
                oom_count=0,
                allocator_specific_stats={"padding_waste_bytes": 0.0},
            )
        )
        stems.append(f"jamba_1_5_mini__tb{tb_human}__seed1")
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)
    out = tmp_path / "frag.png"
    # Should not raise; should pick 4gib internally.
    plot_fragmentation_vs_workload(aggregated, out, model_spec_filter="jamba_1_5_mini")
    assert out.exists()
