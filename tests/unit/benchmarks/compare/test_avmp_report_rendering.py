"""Report rendering tests for the ``avmp_static`` elif branch.

The v1 contract from the planning conversation: AVMP rows fill the same
``kv_free_MiB`` / ``ssm_free_MiB`` columns that ``fixed_dual`` uses,
deriving per-pool free bytes from
``pages_free * (pool_bytes / pages_total)``. ``padding_waste_MiB``
shows ``-`` on AVMP rows. AVMP-specific stats
(``virtual_handles_live``, ``cross_pool_eviction_count``) stay in the
per-cell JSON, not the report table.
"""

from __future__ import annotations

from pathlib import Path

from cachepawl.benchmarks import BenchmarkRun
from cachepawl.benchmarks.compare import (
    SweepConfig,
    aggregate_runs,
    render_markdown_report,
)
from cachepawl.benchmarks.compare.sweep import AllocatorVariant
from tests.unit.benchmarks.compare.conftest import make_run, make_sweep_result

_MIB: float = 1024.0 * 1024.0


def _build_avmp_config(tmp_path: Path) -> SweepConfig:
    return SweepConfig(
        variants=(
            AllocatorVariant("padded_unified", "padded_unified", ()),
            AllocatorVariant("avmp_static_mr05", "avmp_static", (("mamba_ratio", 0.5),)),
        ),
        workload_names=("uniform_short",),
        model_spec_names=("jamba_1_5_mini",),
        total_bytes_options=(1 * 1024**3,),
        device="cpu",
        output_dir=tmp_path,
        seed_replicates=1,
        smoke_num_requests=None,
    )


def _build_runs() -> tuple[list[BenchmarkRun], list[str]]:
    """Two cells: one padded_unified (control), one avmp_static_mr05.

    The AVMP cell carries stats that make the derivation easy to read:
    KV pool has 100 of 200 pages free at 4096 bytes per page (200 MiB
    pool, 50% free => 100 pages_free); SSM pool has 4 of 8 blocks free
    at 1 MiB per block (8 MiB pool).
    """

    runs: list[BenchmarkRun] = []
    stems: list[str] = []

    runs.append(
        make_run(
            allocator_label="padded_unified",
            workload_name="uniform_short",
            seed=1,
            peak_reserved_bytes=1_000_000,
            final_fragmentation=0.05,
            oom_count=0,
            allocator_specific_stats={"padding_waste_bytes": _MIB},
        )
    )
    stems.append("jamba_1_5_mini__tb1gib__seed1")

    runs.append(
        make_run(
            allocator_label="avmp_static_mr05",
            workload_name="uniform_short",
            seed=1,
            peak_reserved_bytes=1_000_000,
            final_fragmentation=0.02,
            oom_count=0,
            allocator_specific_stats={
                "kv_pages_total": 200.0,
                "kv_pages_used": 100.0,
                "kv_pages_free": 100.0,
                "ssm_blocks_total": 8.0,
                "ssm_blocks_used": 4.0,
                "ssm_blocks_free": 4.0,
                "virtual_handles_live": 104.0,
                "cross_pool_eviction_count": 0.0,
                "kv_pool_bytes": 200.0 * 4096.0,
                "ssm_pool_bytes": 8.0 * _MIB,
                "mamba_ratio": 0.5,
            },
        )
    )
    stems.append("jamba_1_5_mini__tb1gib__seed1")

    return runs, stems


def _avmp_row(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("| avmp_static_mr05 "):
            return line
    raise AssertionError("avmp_static_mr05 row not found in report")


def _split_cells(row: str) -> list[str]:
    parts = [cell.strip() for cell in row.split("|")]
    # Leading and trailing splits are empty because the row starts/ends with "|".
    return parts[1:-1]


def test_avmp_row_populates_kv_and_ssm_free_columns(tmp_path: Path) -> None:
    config = _build_avmp_config(tmp_path)
    runs, stems = _build_runs()
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)

    output = tmp_path / "report.md"
    render_markdown_report(
        aggregated,
        output,
        git_sha="0" * 40,
        run_date="2026-05-15",
        hardware_label="cpu (test)",
    )
    text = output.read_text()
    row = _avmp_row(text)
    cells = _split_cells(row)
    # Column order from _render_workload_table:
    # variant | model_spec | total_bytes | peak_reserved_MiB |
    # frag_load | frag_peak | p50 | p99 | oom | padding_waste | kv_free | ssm_free
    padding_waste_text = cells[9]
    kv_free_text = cells[10]
    ssm_free_text = cells[11]

    assert padding_waste_text == "-"
    assert kv_free_text != "-"
    assert ssm_free_text != "-"


def test_avmp_row_kv_free_matches_derived_formula(tmp_path: Path) -> None:
    """kv_free_MiB == kv_pages_free * (kv_pool_bytes / kv_pages_total) / MiB."""

    config = _build_avmp_config(tmp_path)
    runs, stems = _build_runs()
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)

    output = tmp_path / "report.md"
    render_markdown_report(
        aggregated,
        output,
        git_sha="0" * 40,
        run_date="2026-05-15",
        hardware_label="cpu (test)",
    )
    text = output.read_text()
    row = _avmp_row(text)
    cells = _split_cells(row)

    # 100 pages_free * 4096 bytes_per_page = 409600 bytes ≈ 0.391 MiB.
    expected_kv_mib = (100.0 * 4096.0) / _MIB
    assert cells[10] == f"{expected_kv_mib:.3f}"

    # 4 blocks_free * 1 MiB per block = 4 MiB.
    assert cells[11] == "4.000"


def test_avmp_row_handles_zero_pages_total_gracefully(tmp_path: Path) -> None:
    """Pathological row with zero totals must produce 0.000, not a division error."""

    config = _build_avmp_config(tmp_path)

    runs: list[BenchmarkRun] = [
        make_run(
            allocator_label="padded_unified",
            workload_name="uniform_short",
            seed=1,
            peak_reserved_bytes=1_000_000,
            final_fragmentation=0.05,
            oom_count=0,
            allocator_specific_stats={"padding_waste_bytes": 0.0},
        ),
        make_run(
            allocator_label="avmp_static_mr05",
            workload_name="uniform_short",
            seed=1,
            peak_reserved_bytes=1_000_000,
            final_fragmentation=0.05,
            oom_count=0,
            allocator_specific_stats={
                "kv_pages_total": 0.0,
                "kv_pages_used": 0.0,
                "kv_pages_free": 0.0,
                "ssm_blocks_total": 0.0,
                "ssm_blocks_used": 0.0,
                "ssm_blocks_free": 0.0,
                "virtual_handles_live": 0.0,
                "cross_pool_eviction_count": 0.0,
                "kv_pool_bytes": 0.0,
                "ssm_pool_bytes": 0.0,
                "mamba_ratio": 0.5,
            },
        ),
    ]
    stems = ["jamba_1_5_mini__tb1gib__seed1"] * 2
    result = make_sweep_result(config=config, runs=runs, cell_stems=stems)
    aggregated = aggregate_runs(result)

    output = tmp_path / "report.md"
    render_markdown_report(
        aggregated,
        output,
        git_sha="0" * 40,
        run_date="2026-05-15",
        hardware_label="cpu (test)",
    )
    text = output.read_text()
    row = _avmp_row(text)
    cells = _split_cells(row)
    assert cells[10] == "0.000"
    assert cells[11] == "0.000"
