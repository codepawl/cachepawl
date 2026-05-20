"""Generate paper tables from committed sweep aggregated.json files.

Each table is a self-contained ``\\begin{tabular}{...}...\\end{tabular}``
block using booktabs rules. No surrounding ``\\begin{table}`` wrapper -
that is the section file's job, so ``\\input{...}`` slots cleanly into
the author's chosen ``\\begin{table*}[t]`` layout.

Tables produced:

1. ``table_baseline_comparison``: cross-workload OOMs for the 4
   headline variants.
2. ``table_per_workload_winner``: per-workload goodput (req/s) for
   ``fixed_dual_mr05`` vs ``avmp_dynamic_b128`` plus their ratio,
   matching the Section 5.3 prose narrative.
3. ``table_parameter_defaults``: AVMP knob defaults read live from
   :class:`AsymmetricVirtualPool` via :func:`inspect.signature` so the
   table never drifts from the code.
4. ``table_stage1_batchsize``: stage 1 batch size sweep, one row per
   batch_size in 1..256.
5. ``table_stage2_threshold``: stage 2 threshold sweep + stage 1 b128
   reference row, showing the cross-variant tie at 510.

All numerical content comes from
``benchmarks/results/avmp-v2-batchsize-sweep/aggregated.json``,
``benchmarks/results/avmp-v2-threshold-sweep/aggregated.json``, and
``benchmarks/results/avmp-v2-throughput/full/aggregated.json`` plus the
live :class:`AsymmetricVirtualPool` constructor defaults. No hardcoded
data values.
"""

from __future__ import annotations

import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from cachepawl.allocator.avmp import AsymmetricVirtualPool

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
BATCHSIZE_SWEEP: Path = REPO_ROOT / "benchmarks/results/avmp-v2-batchsize-sweep/aggregated.json"
THRESHOLD_SWEEP: Path = REPO_ROOT / "benchmarks/results/avmp-v2-threshold-sweep/aggregated.json"
THROUGHPUT_SWEEP: Path = REPO_ROOT / "benchmarks/results/avmp-v2-throughput/full/aggregated.json"

_HEADLINE_VARIANTS: tuple[str, ...] = (
    "padded_unified",
    "fixed_dual_mr05",
    "avmp_static_mr05",
    "avmp_dynamic_b128",
)

_WORKLOAD_ORDER: tuple[str, ...] = ("uniform_short", "mixed_long", "agentic_burst")
_BATCH_SIZE_SERIES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

_THRESHOLD_TABLE_VARIANTS: tuple[str, ...] = (
    "avmp_dynamic_b128",
    "avmp_dynamic_b128_th_high_010",
    "avmp_dynamic_b128_th_high_020",
    "avmp_dynamic_b128_th_low_002",
    "avmp_dynamic_b128_th_low_010",
)


@dataclass(frozen=True)
class Row:
    variant_label: str
    workload_name: str
    model_spec_name: str
    total_bytes: int
    oom_count_mean: float
    oom_count_std: float
    rebalance_count_median: float
    threshold_low_median: float
    threshold_high_median: float
    goodput_req_per_s_median: float


def _coerce_row(raw: object) -> Row:
    if not isinstance(raw, dict):
        raise TypeError(f"row entry is not a dict: {type(raw).__name__}")
    asp_raw = raw.get("allocator_specific_median", {})
    asp: dict[str, float] = {}
    if isinstance(asp_raw, dict):
        for k, v in asp_raw.items():
            if isinstance(v, (int, float)):
                asp[str(k)] = float(v)
    gp_raw = raw.get("goodput_requests_per_second_median")
    goodput = float(gp_raw) if isinstance(gp_raw, (int, float)) else 0.0
    oom_std_raw = raw.get("oom_count_std")
    oom_std = float(oom_std_raw) if isinstance(oom_std_raw, (int, float)) else 0.0
    return Row(
        variant_label=str(raw["variant_label"]),
        workload_name=str(raw["workload_name"]),
        model_spec_name=str(raw["model_spec_name"]),
        total_bytes=int(cast(int, raw["total_bytes"])),
        oom_count_mean=float(cast(float, raw["oom_count_mean"])),
        oom_count_std=oom_std,
        rebalance_count_median=float(asp.get("rebalance_count", 0.0)),
        threshold_low_median=float(asp.get("threshold_low", 0.0)),
        threshold_high_median=float(asp.get("threshold_high", 0.0)),
        goodput_req_per_s_median=goodput,
    )


def _load_rows(path: Path) -> list[Row]:
    if not path.exists():
        raise FileNotFoundError(f"aggregated.json not found: {path}")
    data: object = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top-level JSON must be an object")
    rows_raw = data.get("rows")
    if not isinstance(rows_raw, list):
        raise TypeError(f"{path}: 'rows' must be a list")
    return [_coerce_row(r) for r in rows_raw]


def _sum_oom_per_workload(rows: list[Row], variant: str) -> dict[str, float]:
    out: dict[str, float] = dict.fromkeys(_WORKLOAD_ORDER, 0.0)
    for r in rows:
        if r.variant_label == variant and r.workload_name in out:
            out[r.workload_name] += r.oom_count_mean
    return out


def _sum_oom_with_std_per_workload(rows: list[Row], variant: str) -> dict[str, tuple[float, float]]:
    """Sum mean OOMs and propagate std across cells assuming independence.

    Per-cell ``oom_count_std`` is taken from :mod:`aggregate`; the std of
    a sum of independent random variables is ``sqrt(sum(var_i))``, which
    is the standard error propagation result. This is not data
    fabrication: it is a derivable transformation of the existing
    pipeline output.
    """

    sums: dict[str, float] = dict.fromkeys(_WORKLOAD_ORDER, 0.0)
    vars_: dict[str, float] = dict.fromkeys(_WORKLOAD_ORDER, 0.0)
    for r in rows:
        if r.variant_label == variant and r.workload_name in sums:
            sums[r.workload_name] += r.oom_count_mean
            vars_[r.workload_name] += r.oom_count_std * r.oom_count_std
    return {w: (sums[w], math.sqrt(vars_[w])) for w in _WORKLOAD_ORDER}


def _sum_rebalance(rows: list[Row], variant: str) -> float:
    return sum(r.rebalance_count_median for r in rows if r.variant_label == variant)


def _mean_goodput_per_workload(rows: list[Row], variant: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {w: [] for w in _WORKLOAD_ORDER}
    for r in rows:
        if r.variant_label == variant and r.workload_name in buckets:
            buckets[r.workload_name].append(r.goodput_req_per_s_median)
    return {w: (sum(v) / len(v) if v else 0.0) for w, v in buckets.items()}


def _write_table(out_dir: Path, basename: str, body: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{basename}.tex"
    path.write_text(body, encoding="utf-8")
    return path


def _tex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def _format_with_rank(values: list[float], lower_is_better: bool, fmt: str) -> list[str]:
    """Format values, wrapping best in \\textbf and second-best in \\underline.

    Ties at the best are all bolded and no second-best is marked. Ties at
    the second-best are all underlined. ``fmt`` is a printf-style spec
    applied to each value (e.g. ``"{:.1f}"``).
    """

    if not values:
        return []
    extreme = min(values) if lower_is_better else max(values)
    best_indices = {i for i, v in enumerate(values) if v == extreme}
    rest = [(i, v) for i, v in enumerate(values) if i not in best_indices]
    second_indices: set[int] = set()
    if rest:
        second_extreme = min(v for _, v in rest) if lower_is_better else max(v for _, v in rest)
        second_indices = {i for i, v in rest if v == second_extreme}
    out: list[str] = []
    for i, v in enumerate(values):
        cell = fmt.format(v)
        if i in best_indices and len(best_indices) < len(values):
            cell = f"\\textbf{{{cell}}}"
        elif i in second_indices:
            cell = f"\\underline{{{cell}}}"
        out.append(cell)
    return out


def table_baseline_comparison(rows: list[Row], out_dir: Path) -> Path:
    per_workload_mean: dict[str, list[float]] = {w: [] for w in _WORKLOAD_ORDER}
    per_workload_std: dict[str, list[float]] = {w: [] for w in _WORKLOAD_ORDER}
    totals: list[float] = []
    total_stds: list[float] = []
    for variant in _HEADLINE_VARIANTS:
        per = _sum_oom_with_std_per_workload(rows, variant)
        for w in _WORKLOAD_ORDER:
            per_workload_mean[w].append(per[w][0])
            per_workload_std[w].append(per[w][1])
        totals.append(sum(per[w][0] for w in _WORKLOAD_ORDER))
        total_stds.append(math.sqrt(sum(per[w][1] ** 2 for w in _WORKLOAD_ORDER)))
    formatted_per_workload = {
        w: _format_with_rank(per_workload_mean[w], lower_is_better=True, fmt="{:.1f}")
        for w in _WORKLOAD_ORDER
    }
    formatted_totals = _format_with_rank(totals, lower_is_better=True, fmt="{:.1f}")
    baseline_total = totals[0]
    delta_cells: list[str] = []
    for i, total in enumerate(totals):
        if i == 0:
            delta_cells.append("---")
            continue
        if baseline_total <= 0.0:
            delta_cells.append("n/a")
            continue
        delta_pct = (total - baseline_total) / baseline_total * 100.0
        sign = "$-$" if delta_pct < 0 else "$+$"
        delta_cells.append(f"{sign}{abs(delta_pct):.1f}\\%")
    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Variant & uniform\\_short & mixed\\_long & agentic\\_burst & Total & "
        "$\\Delta\\%$ vs padded\\_unified \\\\"
    )
    lines.append("\\midrule")
    for i, variant in enumerate(_HEADLINE_VARIANTS):
        cells = [
            f"{formatted_per_workload[w][i]} $\\pm$ {per_workload_std[w][i]:.1f}"
            for w in _WORKLOAD_ORDER
        ]
        total_cell = f"{formatted_totals[i]} $\\pm$ {total_stds[i]:.1f}"
        lines.append(
            f"{_tex_escape(variant)} & {cells[0]} & {cells[1]} & "
            f"{cells[2]} & {total_cell} & {delta_cells[i]} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return _write_table(out_dir, "table_baseline_comparison", "\n".join(lines) + "\n")


def table_per_workload_winner(rows: list[Row], out_dir: Path) -> Path:
    baseline = "fixed_dual_mr05"
    target = "avmp_dynamic_b128"
    baseline_gp = _mean_goodput_per_workload(rows, baseline)
    target_gp = _mean_goodput_per_workload(rows, target)
    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append(
        f"Workload & {_tex_escape(baseline)} (req/s, $\\uparrow$) & "
        f"{_tex_escape(target)} (req/s, $\\uparrow$) & Ratio \\\\"
    )
    lines.append("\\midrule")
    for workload in _WORKLOAD_ORDER:
        base_v = baseline_gp[workload]
        targ_v = target_gp[workload]
        ratio_str = f"{targ_v / base_v:.2f}$\\times$" if base_v > 0.0 else "n/a"
        formatted = _format_with_rank([base_v, targ_v], lower_is_better=False, fmt="{:.2f}")
        lines.append(
            f"{_tex_escape(workload)} & {formatted[0]} & {formatted[1]} & {ratio_str} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return _write_table(out_dir, "table_per_workload_winner", "\n".join(lines) + "\n")


def table_parameter_defaults(out_dir: Path) -> Path:
    sig = inspect.signature(AsymmetricVirtualPool.__init__)
    kept = (
        "mamba_ratio",
        "rebalance_enabled",
        "threshold_low",
        "threshold_high",
        "migration_batch_size",
        "min_rebalance_interval_ops",
    )
    lines: list[str] = []
    lines.append("\\begin{tabular}{ll}")
    lines.append("\\toprule")
    lines.append("Parameter & Constructor default \\\\")
    lines.append("\\midrule")
    for name in kept:
        param = sig.parameters.get(name)
        if param is None or param.default is inspect.Parameter.empty:
            value_str = "(no default)"
        else:
            value_str = repr(param.default)
        lines.append(f"{_tex_escape(name)} & {_tex_escape(value_str)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return _write_table(out_dir, "table_parameter_defaults", "\n".join(lines) + "\n")


def table_stage1_batchsize(rows: list[Row], out_dir: Path) -> Path:
    per_workload_mean: dict[str, list[float]] = {w: [] for w in _WORKLOAD_ORDER}
    per_workload_std: dict[str, list[float]] = {w: [] for w in _WORKLOAD_ORDER}
    totals: list[float] = []
    total_stds: list[float] = []
    for bs in _BATCH_SIZE_SERIES:
        per = _sum_oom_with_std_per_workload(rows, f"avmp_dynamic_b{bs}")
        for w in _WORKLOAD_ORDER:
            per_workload_mean[w].append(per[w][0])
            per_workload_std[w].append(per[w][1])
        totals.append(sum(per[w][0] for w in _WORKLOAD_ORDER))
        total_stds.append(math.sqrt(sum(per[w][1] ** 2 for w in _WORKLOAD_ORDER)))
    formatted_per_workload = {
        w: _format_with_rank(per_workload_mean[w], lower_is_better=True, fmt="{:.1f}")
        for w in _WORKLOAD_ORDER
    }
    formatted_totals = _format_with_rank(totals, lower_is_better=True, fmt="{:.1f}")
    lines: list[str] = []
    lines.append("\\begin{tabular}{rrrrr}")
    lines.append("\\toprule")
    lines.append("batch\\_size & uniform\\_short & mixed\\_long & agentic\\_burst & Total \\\\")
    lines.append("\\midrule")
    for i, bs in enumerate(_BATCH_SIZE_SERIES):
        cells = [
            f"{formatted_per_workload[w][i]} $\\pm$ {per_workload_std[w][i]:.1f}"
            for w in _WORKLOAD_ORDER
        ]
        total_cell = f"{formatted_totals[i]} $\\pm$ {total_stds[i]:.1f}"
        lines.append(f"{bs} & {cells[0]} & {cells[1]} & {cells[2]} & {total_cell} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return _write_table(out_dir, "table_stage1_batchsize", "\n".join(lines) + "\n")


def _representative_thresholds(rows: list[Row], variant: str) -> tuple[float, float]:
    for r in rows:
        if r.variant_label == variant:
            return (r.threshold_low_median, r.threshold_high_median)
    return (0.0, 0.0)


def table_stage2_threshold(
    threshold_rows: list[Row], batchsize_rows: list[Row], out_dir: Path
) -> Path:
    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Variant & threshold\\_low & threshold\\_high & "
        "total\\_oom ($\\downarrow$) & rebalance\\_count \\\\"
    )
    lines.append("\\midrule")
    for variant in _THRESHOLD_TABLE_VARIANTS:
        source = batchsize_rows if variant == "avmp_dynamic_b128" else threshold_rows
        per_with_std = _sum_oom_with_std_per_workload(source, variant)
        total = sum(per_with_std[w][0] for w in _WORKLOAD_ORDER)
        total_std = math.sqrt(sum(per_with_std[w][1] ** 2 for w in _WORKLOAD_ORDER))
        rebal = _sum_rebalance(source, variant)
        th_low, th_high = _representative_thresholds(source, variant)
        lines.append(
            f"{_tex_escape(variant)} & {th_low:.2f} & {th_high:.2f} & "
            f"{total:.1f} $\\pm$ {total_std:.1f} & {rebal:.0f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return _write_table(out_dir, "table_stage2_threshold", "\n".join(lines) + "\n")


def generate_all(out_dir: Path) -> list[Path]:
    batchsize_rows = _load_rows(BATCHSIZE_SWEEP)
    threshold_rows = _load_rows(THRESHOLD_SWEEP)
    throughput_rows = _load_rows(THROUGHPUT_SWEEP)
    return [
        table_baseline_comparison(batchsize_rows, out_dir),
        table_per_workload_winner(throughput_rows, out_dir),
        table_parameter_defaults(out_dir),
        table_stage1_batchsize(batchsize_rows, out_dir),
        table_stage2_threshold(threshold_rows, batchsize_rows, out_dir),
    ]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out_dir = Path(args[0]) if args else REPO_ROOT / "research/avmp/tables/generated"
    paths = generate_all(out_dir)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
