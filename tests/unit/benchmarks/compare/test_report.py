"""Markdown report and JSON summary rendering."""

from __future__ import annotations

import json
from pathlib import Path

from cachepawl.benchmarks import BenchmarkRun
from cachepawl.benchmarks.compare import (
    SweepConfig,
    aggregate_runs,
    render_deterministic_summary,
    render_json_summary,
    render_markdown_report,
)
from cachepawl.benchmarks.compare.sweep import AllocatorVariant
from tests.unit.benchmarks.compare.conftest import make_run, make_sweep_result


def _build_two_variant_two_workload_sweep(tmp_path: Path) -> SweepConfig:
    return SweepConfig(
        variants=(
            AllocatorVariant("padded_unified", "padded_unified", ()),
            AllocatorVariant("fixed_dual_mr05", "fixed_dual", (("mamba_ratio", 0.5),)),
        ),
        workload_names=("uniform_short", "mixed_long"),
        model_spec_names=("jamba_1_5_mini",),
        total_bytes_options=(1 * 1024**3,),
        device="cpu",
        output_dir=tmp_path,
        seed_replicates=2,
        smoke_num_requests=None,
    )


def _build_runs() -> tuple[list[BenchmarkRun], list[str]]:
    runs: list[BenchmarkRun] = []
    stems: list[str] = []
    for variant_label, allocator_kind in (
        ("padded_unified", "padded"),
        ("fixed_dual_mr05", "fixed_dual"),
    ):
        for workload_name in ("uniform_short", "mixed_long"):
            for seed in (1, 2):
                specific: dict[str, float]
                if allocator_kind == "padded":
                    specific = {"padding_waste_bytes": 1024.0 * 1024.0}
                else:
                    specific = {
                        "pool_underused_bytes_kv": 512.0 * 1024.0,
                        "pool_underused_bytes_ssm": 256.0 * 1024.0,
                    }
                runs.append(
                    make_run(
                        allocator_label=variant_label,
                        workload_name=workload_name,
                        seed=seed,
                        peak_reserved_bytes=1_000_000,
                        final_fragmentation=0.05 if variant_label == "padded_unified" else 0.02,
                        oom_count=0,
                        allocator_specific_stats=specific,
                    )
                )
                stems.append(f"jamba_1_5_mini__tb1gib__seed{seed}")
    return runs, stems


def test_render_markdown_report_contains_required_sections(tmp_path: Path) -> None:
    config = _build_two_variant_two_workload_sweep(tmp_path)
    runs, stems = _build_runs()
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)

    output = tmp_path / "report.md"
    render_markdown_report(
        aggregated,
        output,
        git_sha="0123456789abcdef" * 2,
        run_date="2026-05-14",
        hardware_label="cpu (test linux x86_64)",
    )
    text = output.read_text()
    assert "# Allocator baseline comparison" in text
    assert "## How to read" in text
    assert "## Workload: uniform_short" in text
    assert "## Workload: mixed_long" in text
    assert "padded_unified" in text
    assert "fixed_dual_mr05" in text
    assert "padding_waste_MiB" in text
    assert "kv_underused_MiB" in text
    assert "ssm_underused_MiB" in text
    assert "git SHA 0123456789ab" in text  # short SHA (12 chars)
    assert "2026-05-14" in text
    assert "+-" in text  # ASCII +-, not unicode +/-


def test_report_uses_relative_paths_only(tmp_path: Path) -> None:
    """Markdown report must not leak absolute paths or hostnames."""

    config = _build_two_variant_two_workload_sweep(tmp_path)
    runs, stems = _build_runs()
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)
    output = tmp_path / "report.md"
    render_markdown_report(
        aggregated,
        output,
        git_sha="abcdef" * 7,
        run_date="2026-05-14",
        hardware_label="cpu (test)",
    )
    text = output.read_text()
    assert str(tmp_path) not in text
    assert "/home/" not in text
    assert "/Users/" not in text


def test_render_json_summary_is_sorted(tmp_path: Path) -> None:
    """JSON summary must serialize with sort_keys for diff stability."""

    config = _build_two_variant_two_workload_sweep(tmp_path)
    runs, stems = _build_runs()
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)
    output = tmp_path / "aggregated.json"
    render_json_summary(aggregated, output)
    parsed = json.loads(output.read_text())
    assert "rows" in parsed
    # Keys in the first row must be in sorted order. Compare the raw
    # key sequence from json.load (Python preserves insertion order from
    # the file) against its sorted permutation.
    first_row_keys = list(parsed["rows"][0].keys())
    assert first_row_keys == sorted(first_row_keys)


def test_deterministic_summary_omits_latency(tmp_path: Path) -> None:
    """The byte-comparable subset must NOT carry allocate_p99 etc."""

    config = _build_two_variant_two_workload_sweep(tmp_path)
    runs, stems = _build_runs()
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)
    output = tmp_path / "aggregated_deterministic.json"
    render_deterministic_summary(aggregated, output)
    parsed = json.loads(output.read_text())
    assert "rows" in parsed
    for row in parsed["rows"]:
        assert "allocate_p50_ns_median" not in row
        assert "allocate_p95_ns_median" not in row
        assert "allocate_p99_ns_median" not in row
        # but the deterministic fields must remain
        assert "peak_reserved_bytes_mean" in row
        assert "fragmentation_final_mean" in row
        assert "allocator_specific_median" in row
