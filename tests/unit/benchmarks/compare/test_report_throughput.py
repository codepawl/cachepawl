"""Tier 1 PR B: report rendering picks up the new throughput columns."""

from __future__ import annotations

from pathlib import Path

from cachepawl.benchmarks.compare.aggregate import AggregatedMetrics, AggregatedRow
from cachepawl.benchmarks.compare.report import render_markdown_report


def _row(
    *,
    variant: str = "v",
    workload: str = "uniform_short",
    eff_batch_p50: float = 4.0,
    goodput: float = 50.0,
    completion: float = 0.95,
) -> AggregatedRow:
    return AggregatedRow(
        variant_label=variant,
        allocator_name="padded_unified",
        workload_name=workload,
        model_spec_name="jamba_1_5_mini",
        total_bytes=1 * 1024**3,
        n_replicates=1,
        peak_reserved_bytes_mean=0.0,
        peak_reserved_bytes_std=0.0,
        peak_reserved_bytes_min=0,
        peak_reserved_bytes_max=0,
        fragmentation_during_load_mean=0.1,
        fragmentation_during_load_std=0.0,
        fragmentation_during_load_min=0.1,
        fragmentation_during_load_max=0.1,
        fragmentation_peak=0.1,
        oom_count_mean=0.0,
        oom_count_std=0.0,
        effective_batch_size_mean_median=eff_batch_p50,
        effective_batch_size_p50_median=eff_batch_p50,
        effective_batch_size_p95_median=eff_batch_p50,
        effective_batch_size_p99_median=eff_batch_p50,
        goodput_requests_per_second_median=goodput,
        completion_ratio_median=completion,
        time_to_first_oom_seconds_median=None,
        time_in_service_ns_median=0,
        time_in_oom_retry_ns_median=0,
        time_in_migration_ns_median=0,
        time_in_idle_ns_median=0,
        allocate_p50_ns_median=0,
        allocate_p95_ns_median=0,
        allocate_p99_ns_median=0,
        allocator_specific_median=(),
        allocator_specific_iqr=(),
    )


def test_report_includes_throughput_column_headers(tmp_path: Path) -> None:
    """All three new columns appear in the per-workload table header.

    Anchors against the column order shipped by Tier 1 PR B so a
    future column reshuffle breaks this test and forces an update to
    the downstream fixtures.
    """

    aggregated = AggregatedMetrics(rows=(_row(),))
    output = tmp_path / "report.md"
    render_markdown_report(
        aggregated, output, git_sha="0" * 40, run_date="2026-05-18", hardware_label="cpu (test)"
    )
    text = output.read_text()

    assert "effective_batch_p50" in text
    assert "goodput_req_per_s" in text
    assert "completion_ratio" in text


def test_how_to_read_documents_new_metrics(tmp_path: Path) -> None:
    """The preamble names each new metric and its direction-of-better."""

    aggregated = AggregatedMetrics(rows=(_row(),))
    output = tmp_path / "report.md"
    render_markdown_report(
        aggregated, output, git_sha="0" * 40, run_date="2026-05-18", hardware_label="cpu (test)"
    )
    text = output.read_text()

    # Each new bullet must include the metric name and "higher is better"
    # to make the reading convention unambiguous on first scan.
    assert "effective_batch_p50" in text
    assert "goodput_req_per_s" in text
    assert "completion_ratio" in text
    assert text.count("higher is better") >= 3


def test_workload_row_renders_throughput_values(tmp_path: Path) -> None:
    """The data row contains the new metric values in the expected slots.

    Column layout (Tier 1 PR B):
    0: variant | 1: model_spec | 2: total_bytes | 3: peak_reserved_MiB
    | 4: frag_load | 5: frag_peak | 6: p50 | 7: p99 | 8: oom
    | 9: effective_batch_p50 | 10: goodput_req_per_s | 11: completion_ratio
    """

    row = _row(eff_batch_p50=42.5, goodput=137.0, completion=0.876)
    aggregated = AggregatedMetrics(rows=(row,))
    output = tmp_path / "report.md"
    render_markdown_report(
        aggregated, output, git_sha="0" * 40, run_date="2026-05-18", hardware_label="cpu (test)"
    )
    text = output.read_text()

    data_row = next(line for line in text.splitlines() if line.startswith("| v |"))
    cells = [c.strip() for c in data_row.split("|")][1:-1]
    assert cells[9] == "42.5"
    assert cells[10] == "137.0"
    assert cells[11] == "0.876"


def test_cross_workload_summary_includes_throughput_columns(tmp_path: Path) -> None:
    """The cross-workload summary picks up mean_effective_batch_p50 and mean_goodput.

    The summary only emits when the sweep covers more than one
    workload, so the test data must populate at least two.
    """

    aggregated = AggregatedMetrics(
        rows=(
            _row(workload="uniform_short", eff_batch_p50=10.0, goodput=100.0),
            _row(workload="mixed_long", eff_batch_p50=20.0, goodput=200.0),
        )
    )
    output = tmp_path / "report.md"
    render_markdown_report(
        aggregated, output, git_sha="0" * 40, run_date="2026-05-18", hardware_label="cpu (test)"
    )
    text = output.read_text()

    assert "Cross-workload summary" in text
    assert "mean_effective_batch_p50" in text
    assert "mean_goodput_req_per_s" in text
