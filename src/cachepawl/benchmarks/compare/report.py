"""Markdown report and JSON summary rendering for the comparison sweep.

The markdown report is a human-readable view: per-workload tables of
metrics across variants, plus a relative-improvement section that
compares fixed_dual_* variants against padded_unified. The JSON summary
is the same data, sorted-keys and indent-formatted for deterministic
diffs.

Neither output contains absolute paths or hostnames. The footer prints
the git SHA and a hardware label so the file is self-describing.

ASCII conventions: ``+-`` (not the unicode plus-minus glyph) appears in
all mean / std cells. Em dashes are avoided per repo conventions.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from cachepawl.benchmarks.compare.aggregate import (
    AggregatedMetrics,
    AggregatedRow,
    compute_relative_improvement,
)
from cachepawl.benchmarks.compare.sweep import total_bytes_human

_MIB: int = 1024 * 1024
_NS_PER_US: int = 1000
_BASELINE_VARIANT: str = "padded_unified"


def render_markdown_report(
    aggregated: AggregatedMetrics,
    output_path: Path,
    *,
    git_sha: str,
    run_date: str,
    hardware_label: str,
    regenerate_command: str = (
        "python -m cachepawl.benchmarks.compare --quick --device cpu "
        "--output benchmarks/results/baseline/quick/"
    ),
) -> None:
    """Write the markdown report for ``aggregated`` to ``output_path``."""

    sections: list[str] = []
    sections.append(_render_title())
    sections.append(_render_how_to_read())
    for workload_name in _workloads_in_order(aggregated.rows):
        sections.append(_render_workload_table(workload_name, aggregated.rows))
    rel_section = _render_relative_improvement(aggregated.rows)
    if rel_section:
        sections.append(rel_section)
    sections.append(_render_footer(git_sha, run_date, hardware_label, regenerate_command))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(sections) + "\n")


def render_json_summary(aggregated: AggregatedMetrics, output_path: Path) -> None:
    """Write the JSON summary (deterministic key order) to ``output_path``."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(aggregated.to_dict(), sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    )


def render_deterministic_summary(aggregated: AggregatedMetrics, output_path: Path) -> None:
    """Write the byte-comparable subset (no latency stats) for reproducibility tests."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            aggregated.deterministic_subset(),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
        )
        + "\n"
    )


def _render_title() -> str:
    return "# Allocator baseline comparison"


def _render_how_to_read() -> str:
    lines = [
        "## How to read",
        "",
        "- `peak_reserved_MiB`: max bytes the pool held during the run (lower is better).",
        "- `fragmentation_ratio`: final `1 - allocated/reserved` sample (lower is better).",
        "- `alloc_p50_us` / `alloc_p99_us`: allocate-call latency in microseconds "
        "(latency varies across reruns).",
        "- `oom_count`: number of `OutOfMemoryError` raised during the run (lower is better).",
        "- `padding_waste_MiB`: bytes wasted by padded_unified rounding SSM blocks up to the "
        "KV page size. Only meaningful for `padded_unified` rows.",
        "- `kv_underused_MiB`, `ssm_underused_MiB`: bytes in each fixed_dual pool that were "
        "reserved but unused. Only meaningful for `fixed_dual_*` rows.",
        "- `mean +- std`: mean across replicates with population standard deviation (ddof=0).",
    ]
    return "\n".join(lines)


def _workloads_in_order(rows: Sequence[AggregatedRow]) -> list[str]:
    """Preserve first-seen order so the report layout is stable."""

    seen: list[str] = []
    seen_set: set[str] = set()
    for row in rows:
        if row.workload_name not in seen_set:
            seen.append(row.workload_name)
            seen_set.add(row.workload_name)
    return seen


def _render_workload_table(workload_name: str, rows: Sequence[AggregatedRow]) -> str:
    cells = [r for r in rows if r.workload_name == workload_name]
    if not cells:
        return ""
    header = (
        "| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_ratio | "
        "alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | "
        "kv_underused_MiB | ssm_underused_MiB |"
    )
    divider = "| " + " | ".join(["---"] * 11) + " |"
    body = [_render_row(row) for row in cells]
    lines = [f"## Workload: {workload_name}", "", header, divider, *body]
    return "\n".join(lines)


def _render_row(row: AggregatedRow) -> str:
    tb_label = total_bytes_human(row.total_bytes)
    peak_mib_mean = row.peak_reserved_bytes_mean / _MIB
    peak_mib_std = row.peak_reserved_bytes_std / _MIB
    p50_us = row.allocate_p50_ns_median / _NS_PER_US
    p99_us = row.allocate_p99_ns_median / _NS_PER_US

    stats_map = dict(row.allocator_specific_median)
    if row.allocator_name == "padded_unified":
        padding_waste_mib_text = _format_mib(stats_map.get("padding_waste_bytes", 0.0))
        kv_under_text = "-"
        ssm_under_text = "-"
    elif row.allocator_name == "fixed_dual":
        padding_waste_mib_text = "-"
        kv_under_text = _format_mib(stats_map.get("pool_underused_bytes_kv", 0.0))
        ssm_under_text = _format_mib(stats_map.get("pool_underused_bytes_ssm", 0.0))
    else:
        padding_waste_mib_text = "-"
        kv_under_text = "-"
        ssm_under_text = "-"

    return (
        f"| {row.variant_label} | {row.model_spec_name} | {tb_label} | "
        f"{peak_mib_mean:.2f} +- {peak_mib_std:.2f} | "
        f"{row.fragmentation_final_mean:.3f} +- {row.fragmentation_final_std:.3f} | "
        f"{p50_us:.2f} | {p99_us:.2f} | {row.oom_count_mean:.1f} | "
        f"{padding_waste_mib_text} | {kv_under_text} | {ssm_under_text} |"
    )


def _format_mib(byte_value: float) -> str:
    return f"{byte_value / _MIB:.3f}"


def _render_relative_improvement(rows: Sequence[AggregatedRow]) -> str:
    baseline_rows = [r for r in rows if r.variant_label == _BASELINE_VARIANT]
    if not baseline_rows:
        return ""
    by_variant: dict[str, list[AggregatedRow]] = defaultdict(list)
    for row in rows:
        if row.variant_label == _BASELINE_VARIANT:
            continue
        by_variant[row.variant_label].append(row)
    if not by_variant:
        return ""

    sections = [f"## Relative improvement vs {_BASELINE_VARIANT}", ""]
    for variant_label in sorted(by_variant):
        deltas = compute_relative_improvement(baseline_rows, by_variant[variant_label])
        if not deltas:
            continue
        sections.append(f"### {variant_label}")
        sections.append("")
        sections.append(
            "| workload | model_spec | total_bytes | "
            "fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |"
        )
        sections.append("| --- | --- | --- | --- | --- | --- |")
        for key in sorted(deltas.keys()):
            workload, model_spec, _variant, total_bytes = key
            metrics = deltas[key]
            sections.append(
                f"| {workload} | {model_spec} | {total_bytes_human(total_bytes)} | "
                f"{_format_pct(metrics.get('fragmentation_final_mean', 0.0))} | "
                f"{_format_pct(metrics.get('peak_reserved_bytes_mean', 0.0))} | "
                f"{_format_pct(metrics.get('oom_count_mean', 0.0))} |"
            )
        sections.append("")
    return "\n".join(sections).rstrip()


def _format_pct(value: float) -> str:
    if value == float("inf"):
        return "+inf"
    if value == float("-inf"):
        return "-inf"
    return f"{value:+.2f}%"


def _render_footer(
    git_sha: str,
    run_date: str,
    hardware_label: str,
    regenerate_command: str,
) -> str:
    short_sha = (git_sha[:12]) if git_sha and git_sha != "unknown" else "unknown"
    lines = [
        "---",
        "",
        f"Generated: {run_date} from git SHA {short_sha}, hardware: {hardware_label}.",
        "",
        f"Regenerate: `{regenerate_command}`",
    ]
    return "\n".join(lines)


__all__ = [
    "render_deterministic_summary",
    "render_json_summary",
    "render_markdown_report",
]
