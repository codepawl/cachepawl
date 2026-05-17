"""Unit tests for the lexicographic ranking analyzer."""

from __future__ import annotations

import pytest

from cachepawl.benchmarks.analysis.lexicographic_rank import (
    rank_variants,
)
from cachepawl.benchmarks.compare.aggregate import AggregatedMetrics, AggregatedRow


def _row(
    *,
    variant: str,
    workload: str = "w0",
    oom: float = 0.0,
    frag: float = 0.0,
    peak_bytes: float = 0.0,
) -> AggregatedRow:
    return AggregatedRow(
        variant_label=variant,
        allocator_name="x",
        workload_name=workload,
        model_spec_name="m",
        total_bytes=1024,
        n_replicates=1,
        peak_reserved_bytes_mean=peak_bytes,
        peak_reserved_bytes_std=0.0,
        peak_reserved_bytes_min=int(peak_bytes),
        peak_reserved_bytes_max=int(peak_bytes),
        fragmentation_during_load_mean=frag,
        fragmentation_during_load_std=0.0,
        fragmentation_during_load_min=frag,
        fragmentation_during_load_max=frag,
        fragmentation_peak=frag,
        oom_count_mean=oom,
        oom_count_std=0.0,
        allocate_p50_ns_median=0,
        allocate_p95_ns_median=0,
        allocate_p99_ns_median=0,
        allocator_specific_median=(),
        allocator_specific_iqr=(),
    )


def test_rank_orders_by_oom_then_fragmentation() -> None:
    """Lower oom wins. Within the OOM tie tolerance, lower fragmentation wins."""

    rows = (
        _row(variant="a", oom=100.0, frag=0.4),  # mid oom, low frag
        _row(variant="b", oom=50.0, frag=0.6),  # low oom, high frag
        _row(variant="c", oom=200.0, frag=0.1),  # high oom, very low frag
    )
    rankings = rank_variants(AggregatedMetrics(rows=rows), oom_tie_tolerance=1.0)
    assert [r.variant_label for r in rankings] == ["b", "a", "c"]


def test_rank_ties_broken_by_fragmentation_within_tolerance() -> None:
    """Two variants within `oom_tie_tolerance` of each other tie on OOM and break on frag."""

    rows = (
        _row(variant="hi_frag", oom=10.0, frag=0.9),
        _row(variant="lo_frag", oom=10.5, frag=0.1),  # within tolerance 1.0
    )
    rankings = rank_variants(AggregatedMetrics(rows=rows), oom_tie_tolerance=1.0)
    assert [r.variant_label for r in rankings] == ["lo_frag", "hi_frag"]


def test_rank_aggregates_across_workloads() -> None:
    """A variant with rows in 2 workloads sums OOM and means fragmentation."""

    rows = (
        _row(variant="a", workload="w1", oom=10.0, frag=0.2),
        _row(variant="a", workload="w2", oom=20.0, frag=0.4),
    )
    rankings = rank_variants(AggregatedMetrics(rows=rows))
    assert len(rankings) == 1
    assert rankings[0].cross_workload_total_oom == 30.0
    assert rankings[0].cross_workload_mean_frag_during_load == pytest.approx(0.3)


def test_rank_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        rank_variants(AggregatedMetrics(rows=()))


def test_rank_peak_reserved_aggregated() -> None:
    rows = (
        _row(variant="a", peak_bytes=1024 * 1024 * 100),  # 100 MiB
        _row(variant="a", workload="w2", peak_bytes=1024 * 1024 * 200),  # 200 MiB
    )
    rankings = rank_variants(AggregatedMetrics(rows=rows))
    assert rankings[0].cross_workload_mean_peak_reserved_mib == pytest.approx(150.0)


def test_render_table_includes_variant_labels() -> None:
    from cachepawl.benchmarks.analysis.lexicographic_rank import render_table

    rows = (
        _row(variant="apple", oom=1.0),
        _row(variant="banana", oom=2.0),
    )
    rankings = rank_variants(AggregatedMetrics(rows=rows))
    table = render_table(rankings)
    assert "apple" in table
    assert "banana" in table
    assert table.index("apple") < table.index("banana")


def test_render_table_top_n() -> None:
    from cachepawl.benchmarks.analysis.lexicographic_rank import render_table

    rows = tuple(_row(variant=f"v{i}", oom=float(i)) for i in range(10))
    rankings = rank_variants(AggregatedMetrics(rows=rows))
    table = render_table(rankings, top=3)
    assert "v0" in table
    assert "v2" in table
    assert "v5" not in table
