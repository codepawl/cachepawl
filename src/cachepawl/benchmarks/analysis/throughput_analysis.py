"""Throughput-metric analysis for Tier 1 PR B.

Reads an ``aggregated.json`` produced by the comparison sweep and
emits a markdown analysis report that:

1. Tabulates per-workload effective batch size and goodput for the
   throughput target (default ``avmp_dynamic_b128``) against the
   reference baseline (default ``fixed_dual_mr05``).
2. Evaluates the pre-registered stop rule with explicit PASS / FAIL
   verdict.
3. Surfaces the 3-level lexicographic ranking from
   :mod:`cachepawl.benchmarks.analysis.lexicographic_rank`.

The stop rule (from RFC 0002 / Tier 1 PR B design doc):

    The throughput claim is justified iff either:
        - effective_batch_size_p50 for the target variant >= 1.05x
          the baseline on at least 2 workloads, OR
        - goodput_requests_per_second for the target variant >= 1.10x
          the baseline on at least 1 workload.

    Otherwise the paper reverts to the OOM-only claim.

The script ALWAYS writes the verdict (PASS / FAIL) regardless of
outcome, so the artifact lands deterministically and reviewers do
not need to run the script themselves to find out which branch the
paper takes.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cachepawl.benchmarks.analysis.lexicographic_rank import (
    VariantRanking,
    rank_variants,
    render_table,
)
from cachepawl.benchmarks.compare.aggregate import AggregatedMetrics, AggregatedRow

StopRuleVerdictKind = Literal["PASS_BATCH_SIZE", "PASS_GOODPUT", "FAIL"]


@dataclass(frozen=True, slots=True)
class WorkloadComparison:
    """Per-workload target-vs-baseline ratios for the two throughput metrics."""

    workload_name: str
    target_eff_batch_p50: float
    baseline_eff_batch_p50: float
    eff_batch_ratio: float
    target_goodput: float
    baseline_goodput: float
    goodput_ratio: float


@dataclass(frozen=True, slots=True)
class StopRuleVerdict:
    """Outcome of the pre-registered stop rule."""

    target: str
    baseline: str
    batch_size_workloads_passing: tuple[str, ...]
    goodput_workloads_passing: tuple[str, ...]
    verdict: StopRuleVerdictKind


def evaluate_stop_rule(
    aggregated: AggregatedMetrics,
    *,
    target: str = "avmp_dynamic_b128",
    baseline: str = "fixed_dual_mr05",
    batch_size_ratio: float = 1.05,
    batch_size_min_workloads: int = 2,
    goodput_ratio: float = 1.10,
    goodput_min_workloads: int = 1,
) -> StopRuleVerdict:
    """Apply the pre-registered stop rule to ``aggregated``.

    For each workload present in the sweep, compute the ratio of
    target-variant ``effective_batch_size_p50_median`` (and
    ``goodput_requests_per_second_median``) to the baseline variant.
    A workload "passes" the batch-size criterion if the ratio meets
    ``batch_size_ratio`` (default 1.05). A workload "passes" the
    goodput criterion if the ratio meets ``goodput_ratio`` (default
    1.10). The stop rule fires (returns ``PASS_*``) when either
    bucket reaches its minimum-workload threshold.

    Baseline rows with a zero baseline value yield a ratio of
    ``+inf`` when the target is strictly positive and ``0.0`` when
    both sides are zero, mirroring the convention in
    :func:`cachepawl.benchmarks.compare.aggregate.compute_relative_improvement`.
    """

    comparisons = build_workload_comparisons(aggregated, target=target, baseline=baseline)
    batch_pass = tuple(
        c.workload_name for c in comparisons if c.eff_batch_ratio >= batch_size_ratio
    )
    goodput_pass = tuple(c.workload_name for c in comparisons if c.goodput_ratio >= goodput_ratio)
    if len(batch_pass) >= batch_size_min_workloads:
        verdict: StopRuleVerdictKind = "PASS_BATCH_SIZE"
    elif len(goodput_pass) >= goodput_min_workloads:
        verdict = "PASS_GOODPUT"
    else:
        verdict = "FAIL"
    return StopRuleVerdict(
        target=target,
        baseline=baseline,
        batch_size_workloads_passing=batch_pass,
        goodput_workloads_passing=goodput_pass,
        verdict=verdict,
    )


def build_workload_comparisons(
    aggregated: AggregatedMetrics,
    *,
    target: str,
    baseline: str,
) -> list[WorkloadComparison]:
    """Per-workload target-vs-baseline pairs, aggregated across cells.

    When a workload has multiple cells (e.g. two model_specs x three
    total_bytes), the per-variant value is the arithmetic mean of the
    cell medians. This matches the report's cross-workload summary
    and keeps the stop rule insensitive to the granular cell layout.
    """

    target_cells: dict[str, list[AggregatedRow]] = defaultdict(list)
    baseline_cells: dict[str, list[AggregatedRow]] = defaultdict(list)
    for row in aggregated.rows:
        if row.variant_label == target:
            target_cells[row.workload_name].append(row)
        elif row.variant_label == baseline:
            baseline_cells[row.workload_name].append(row)

    workloads = sorted(set(target_cells) & set(baseline_cells))
    comparisons: list[WorkloadComparison] = []
    for workload in workloads:
        t_rows = target_cells[workload]
        b_rows = baseline_cells[workload]
        t_eff = _mean(r.effective_batch_size_p50_median for r in t_rows)
        b_eff = _mean(r.effective_batch_size_p50_median for r in b_rows)
        t_gp = _mean(r.goodput_requests_per_second_median for r in t_rows)
        b_gp = _mean(r.goodput_requests_per_second_median for r in b_rows)
        comparisons.append(
            WorkloadComparison(
                workload_name=workload,
                target_eff_batch_p50=t_eff,
                baseline_eff_batch_p50=b_eff,
                eff_batch_ratio=_ratio(t_eff, b_eff),
                target_goodput=t_gp,
                baseline_goodput=b_gp,
                goodput_ratio=_ratio(t_gp, b_gp),
            )
        )
    return comparisons


def render_analysis_markdown(
    aggregated: AggregatedMetrics,
    verdict: StopRuleVerdict,
    rankings: list[VariantRanking],
    output_path: Path,
) -> None:
    """Write the throughput-analysis markdown report.

    Sections:

    1. Pre-registered stop rule (explicit PASS / FAIL).
    2. Hypothesis evaluation table (per-workload ratios).
    3. Per-workload throughput numbers (target + baseline absolute values).
    4. 3-level lexicographic ranking.
    """

    comparisons = build_workload_comparisons(
        aggregated, target=verdict.target, baseline=verdict.baseline
    )

    lines: list[str] = []
    lines.append("# Tier 1 PR B throughput analysis")
    lines.append("")
    lines.append(
        f"Target variant: `{verdict.target}`. Baseline for ratio comparison: `{verdict.baseline}`."
    )
    lines.append("")
    lines.append("## Pre-registered stop rule")
    lines.append("")
    lines.append(
        "The throughput claim is justified iff EITHER "
        "`effective_batch_size_p50` >= 1.05x baseline on at least 2 workloads "
        "OR `goodput_requests_per_second` >= 1.10x baseline on at least 1 "
        "workload. The criterion was registered in the design doc before the "
        "sweep was run."
    )
    lines.append("")
    lines.append(f"**Verdict: {verdict.verdict}**")
    lines.append("")
    lines.append(
        f"- Workloads where eff_batch_size_p50 ratio >= 1.05: "
        f"{_join_or_none(verdict.batch_size_workloads_passing)} "
        f"(threshold: 2)"
    )
    lines.append(
        f"- Workloads where goodput ratio >= 1.10: "
        f"{_join_or_none(verdict.goodput_workloads_passing)} "
        f"(threshold: 1)"
    )
    if verdict.verdict == "FAIL":
        lines.append("")
        lines.append(
            "The throughput claim is NOT justified by this sweep. The paper "
            "should report only the OOM-count reduction headline; the "
            "effective_batch_size and goodput numbers may still appear as "
            "secondary evidence but the framing must not assert a throughput "
            "win."
        )
    lines.append("")

    lines.append("## Hypothesis evaluation (per-workload ratios)")
    lines.append("")
    lines.append(
        "| workload | eff_batch_p50 (target) | eff_batch_p50 (baseline) | "
        "ratio | goodput (target) | goodput (baseline) | ratio |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for c in comparisons:
        lines.append(
            f"| {c.workload_name} | "
            f"{c.target_eff_batch_p50:.2f} | {c.baseline_eff_batch_p50:.2f} | "
            f"{_format_ratio(c.eff_batch_ratio)} | "
            f"{c.target_goodput:.2f} | {c.baseline_goodput:.2f} | "
            f"{_format_ratio(c.goodput_ratio)} |"
        )
    lines.append("")

    lines.append("## Cross-workload lexicographic ranking (3-level)")
    lines.append("")
    lines.append(
        "Sort key: `(total_oom asc, effective_batch_size_p50 desc, "
        "fragmentation_during_load asc)`. Lower OOM wins; within "
        "tie-tolerance, higher sustained batch size wins; remaining ties "
        "break on lower fragmentation."
    )
    lines.append("")
    lines.append("```")
    lines.append(render_table(rankings))
    lines.append("```")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _mean(values: Iterable[float]) -> float:
    """Arithmetic mean over an iterable of floats; empty -> 0.0."""

    seq = list(values)
    if not seq:
        return 0.0
    return float(sum(seq)) / len(seq)


def _ratio(target: float, baseline: float) -> float:
    """Target/baseline with safe divide-by-zero behavior.

    Mirrors :func:`_percent_better` in the aggregate module: when the
    baseline is exactly zero AND the target is strictly positive, the
    ratio is ``+inf``; when both are zero the ratio is ``0.0``. This
    keeps stop-rule comparisons total over the rationals.
    """

    if baseline == 0.0:
        if target == 0.0:
            return 0.0
        return float("inf")
    return target / baseline


def _format_ratio(ratio: float) -> str:
    if ratio == float("inf"):
        return "+inf"
    return f"{ratio:.3f}x"


def _join_or_none(items: tuple[str, ...]) -> str:
    if not items:
        return "(none)"
    return ", ".join(items)


def _aggregated_from_json_path(path: Path) -> AggregatedMetrics:
    """Load an ``aggregated.json`` into an :class:`AggregatedMetrics`."""

    # Reuse the lex-rank loader so older artifacts (pre-1.2.0) keep
    # working — missing throughput fields default cleanly there.
    from cachepawl.benchmarks.analysis.lexicographic_rank import (
        _aggregated_from_json_path as _loader,
    )

    return _loader(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cachepawl.benchmarks.analysis.throughput_analysis",
        description=(
            "Evaluate the Tier 1 PR B throughput stop rule against an "
            "aggregated.json and write a markdown analysis."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="path to an aggregated.json file produced by the comparison sweep",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="path to write the throughput_analysis.md report",
    )
    parser.add_argument(
        "--target",
        default="avmp_dynamic_b128",
        help="variant label whose throughput is being evaluated (default: avmp_dynamic_b128)",
    )
    parser.add_argument(
        "--baseline",
        default="fixed_dual_mr05",
        help="baseline variant for ratio comparison (default: fixed_dual_mr05)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    aggregated = _aggregated_from_json_path(args.input)
    verdict = evaluate_stop_rule(aggregated, target=args.target, baseline=args.baseline)
    rankings = rank_variants(aggregated)
    render_analysis_markdown(aggregated, verdict, rankings, args.output)
    print(f"wrote throughput analysis: {args.output} (verdict: {verdict.verdict})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


__all__ = [
    "StopRuleVerdict",
    "StopRuleVerdictKind",
    "WorkloadComparison",
    "build_workload_comparisons",
    "evaluate_stop_rule",
    "main",
    "render_analysis_markdown",
]


def _load_aggregated_via_dict(data: dict[str, object]) -> AggregatedMetrics:
    """Build an :class:`AggregatedMetrics` from an already-parsed dict.

    Public for tests so they can construct synthetic aggregated data
    without round-tripping through the filesystem. Reuses the lex-rank
    row loader to inherit backward-compat defaults.
    """

    from cachepawl.benchmarks.analysis.lexicographic_rank import _row_from_dict

    rows_data = data["rows"]
    if not isinstance(rows_data, list):
        raise ValueError(f"expected 'rows' to be a list, got {type(rows_data).__name__}")
    rows = tuple(_row_from_dict(r) for r in rows_data)
    return AggregatedMetrics(rows=rows)
