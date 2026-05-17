"""Lexicographic variant ranking for parameter-sweep analysis.

Given an :class:`AggregatedMetrics` (or its JSON form), rank each variant
by ``(cross_workload_total_oom, cross_workload_mean_fragmentation_during_load)``
with a configurable OOM tie-tolerance: variants within ``oom_tie_tolerance``
of the leader's OOM count are sorted by fragmentation instead. The
ranking is used in the v2 stage-1 batch_size sweep PR body to identify
the best variant.

This module is purely an analyzer. It reads aggregated artifacts;
nothing it does affects the allocator code path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from cachepawl.benchmarks.compare.aggregate import AggregatedMetrics, AggregatedRow

_MIB: float = 1024.0 * 1024.0


@dataclass(frozen=True, slots=True)
class VariantRanking:
    """One row of the ranked output."""

    variant_label: str
    cross_workload_total_oom: float
    cross_workload_mean_frag_during_load: float
    cross_workload_mean_peak_reserved_mib: float


def rank_variants(
    aggregated: AggregatedMetrics,
    *,
    oom_tie_tolerance: float = 1.0,
) -> list[VariantRanking]:
    """Rank variants best-first by ``(total_oom asc, mean_fragmentation asc)``.

    Within ``oom_tie_tolerance`` of the lowest OOM count, ties break on
    lower mean fragmentation_during_load. Empty input raises ``ValueError``.

    The cross-workload aggregation here mirrors the report.md cross-workload
    summary: total_oom is the sum of per-cell oom_count_mean, fragmentation
    is the arithmetic mean of per-cell fragmentation_during_load_mean.
    """

    if not aggregated.rows:
        raise ValueError("aggregated.rows is empty; nothing to rank")

    by_variant: dict[str, list[AggregatedRow]] = {}
    for row in aggregated.rows:
        by_variant.setdefault(row.variant_label, []).append(row)

    rankings = [
        VariantRanking(
            variant_label=label,
            cross_workload_total_oom=sum(r.oom_count_mean for r in rows),
            cross_workload_mean_frag_during_load=statistics.mean(
                r.fragmentation_during_load_mean for r in rows
            ),
            cross_workload_mean_peak_reserved_mib=statistics.mean(
                r.peak_reserved_bytes_mean / _MIB for r in rows
            ),
        )
        for label, rows in by_variant.items()
    ]

    # Sort by (oom, frag) with tie-tolerance bucket on oom.
    rankings.sort(
        key=lambda r: (
            round(r.cross_workload_total_oom / max(oom_tie_tolerance, 1e-9)),
            r.cross_workload_mean_frag_during_load,
        )
    )
    return rankings


def render_table(rankings: Iterable[VariantRanking], top: int | None = None) -> str:
    """Render the ranking as a fixed-width text table for the PR body."""

    rows = list(rankings)
    if top is not None:
        rows = rows[:top]
    if not rows:
        return "(no rankings to display)"
    lines = [
        f"{'rank':>4}  {'variant_label':<24}  {'total_oom':>10}  {'mean_frag':>9}  {'peak_MiB':>9}",
        f"{'-' * 4}  {'-' * 24}  {'-' * 10}  {'-' * 9}  {'-' * 9}",
    ]
    for i, r in enumerate(rows, start=1):
        lines.append(
            f"{i:>4}  {r.variant_label:<24}  {r.cross_workload_total_oom:>10.1f}  "
            f"{r.cross_workload_mean_frag_during_load:>9.3f}  "
            f"{r.cross_workload_mean_peak_reserved_mib:>9.0f}"
        )
    return "\n".join(lines)


def _aggregated_from_json_path(path: Path) -> AggregatedMetrics:
    """Load an ``aggregated.json`` file into an :class:`AggregatedMetrics`."""

    data = json.loads(path.read_text())
    rows_data = data["rows"]
    if not isinstance(rows_data, list):
        raise ValueError(f"{path}: expected 'rows' to be a list, got {type(rows_data).__name__}")
    rows = tuple(_row_from_dict(r) for r in rows_data)
    return AggregatedMetrics(rows=rows)


def _row_from_dict(row: object) -> AggregatedRow:
    if not isinstance(row, dict):
        raise ValueError(f"each row must be a dict, got {type(row).__name__}")
    return AggregatedRow(
        variant_label=str(row["variant_label"]),
        allocator_name=str(row["allocator_name"]),
        workload_name=str(row["workload_name"]),
        model_spec_name=str(row["model_spec_name"]),
        total_bytes=int(row["total_bytes"]),
        n_replicates=int(row["n_replicates"]),
        peak_reserved_bytes_mean=float(row["peak_reserved_bytes_mean"]),
        peak_reserved_bytes_std=float(row["peak_reserved_bytes_std"]),
        peak_reserved_bytes_min=int(row["peak_reserved_bytes_min"]),
        peak_reserved_bytes_max=int(row["peak_reserved_bytes_max"]),
        fragmentation_during_load_mean=float(row["fragmentation_during_load_mean"]),
        fragmentation_during_load_std=float(row["fragmentation_during_load_std"]),
        fragmentation_during_load_min=float(row["fragmentation_during_load_min"]),
        fragmentation_during_load_max=float(row["fragmentation_during_load_max"]),
        fragmentation_peak=float(row["fragmentation_peak"]),
        oom_count_mean=float(row["oom_count_mean"]),
        oom_count_std=float(row["oom_count_std"]),
        allocate_p50_ns_median=int(row["allocate_p50_ns_median"]),
        allocate_p95_ns_median=int(row["allocate_p95_ns_median"]),
        allocate_p99_ns_median=int(row["allocate_p99_ns_median"]),
        allocator_specific_median=tuple(
            (str(k), float(v)) for k, v in row["allocator_specific_median"].items()
        ),
        allocator_specific_iqr=tuple(
            (str(k), float(v)) for k, v in row["allocator_specific_iqr"].items()
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cachepawl.benchmarks.analysis.lexicographic_rank",
        description=(
            "Rank variants in an aggregated.json file by (total_oom asc, mean_fragmentation asc)."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="path to an aggregated.json file (e.g. benchmarks/results/.../aggregated.json)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="show only the top N variants (default: all)",
    )
    parser.add_argument(
        "--oom-tie-tolerance",
        type=float,
        default=1.0,
        help=(
            "variants within this many OOMs of the leader are sorted "
            "by fragmentation (default: 1.0)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    aggregated = _aggregated_from_json_path(args.input)
    rankings = rank_variants(aggregated, oom_tie_tolerance=args.oom_tie_tolerance)
    print(render_table(rankings, top=args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
