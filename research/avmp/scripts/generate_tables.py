"""Generate paper tables from committed sweep aggregated.json files.

Each table is a self-contained ``\\begin{tabular}{...}...\\end{tabular}``
block using booktabs rules. No surrounding ``\\begin{table}`` wrapper -
that is the section file's job, so ``\\input{...}`` slots cleanly into
the author's chosen ``\\begin{table*}[t]`` layout.

Tables produced:

1. ``table_baseline_comparison``: cross-workload OOMs for the 4
   headline variants.
2. ``table_per_workload_winner``: per-workload best variant + delta
   vs fixed_dual_mr05.
3. ``table_parameter_defaults``: AVMP knob defaults read live from
   :class:`AsymmetricVirtualPool` via :func:`inspect.signature` so the
   table never drifts from the code.
4. ``table_stage1_batchsize``: stage 1 batch size sweep, one row per
   batch_size in 1..256.
5. ``table_stage2_threshold``: stage 2 threshold sweep + stage 1 b128
   reference row, showing the cross-variant tie at 510.

All numerical content comes from
``benchmarks/results/avmp-v2-batchsize-sweep/aggregated.json`` and
``benchmarks/results/avmp-v2-threshold-sweep/aggregated.json`` plus the
live :class:`AsymmetricVirtualPool` constructor defaults. No hardcoded
data values.
"""

from __future__ import annotations

import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from cachepawl.allocator.avmp import AsymmetricVirtualPool

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
BATCHSIZE_SWEEP: Path = REPO_ROOT / "benchmarks/results/avmp-v2-batchsize-sweep/aggregated.json"
THRESHOLD_SWEEP: Path = REPO_ROOT / "benchmarks/results/avmp-v2-threshold-sweep/aggregated.json"

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
    rebalance_count_median: float
    threshold_low_median: float
    threshold_high_median: float


def _coerce_row(raw: object) -> Row:
    if not isinstance(raw, dict):
        raise TypeError(f"row entry is not a dict: {type(raw).__name__}")
    asp_raw = raw.get("allocator_specific_median", {})
    asp: dict[str, float] = {}
    if isinstance(asp_raw, dict):
        for k, v in asp_raw.items():
            if isinstance(v, (int, float)):
                asp[str(k)] = float(v)
    return Row(
        variant_label=str(raw["variant_label"]),
        workload_name=str(raw["workload_name"]),
        model_spec_name=str(raw["model_spec_name"]),
        total_bytes=int(cast(int, raw["total_bytes"])),
        oom_count_mean=float(cast(float, raw["oom_count_mean"])),
        rebalance_count_median=float(asp.get("rebalance_count", 0.0)),
        threshold_low_median=float(asp.get("threshold_low", 0.0)),
        threshold_high_median=float(asp.get("threshold_high", 0.0)),
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


def _sum_rebalance(rows: list[Row], variant: str) -> float:
    return sum(r.rebalance_count_median for r in rows if r.variant_label == variant)


def _write_table(out_dir: Path, basename: str, body: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{basename}.tex"
    path.write_text(body, encoding="utf-8")
    return path


def _tex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def table_baseline_comparison(rows: list[Row], out_dir: Path) -> Path:
    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Variant & uniform\\_short & mixed\\_long & agentic\\_burst & Total \\\\")
    lines.append("\\midrule")
    for variant in _HEADLINE_VARIANTS:
        per = _sum_oom_per_workload(rows, variant)
        total = sum(per.values())
        lines.append(
            f"{_tex_escape(variant)} & {per['uniform_short']:.1f} & "
            f"{per['mixed_long']:.1f} & {per['agentic_burst']:.1f} & "
            f"{total:.1f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return _write_table(out_dir, "table_baseline_comparison", "\n".join(lines) + "\n")


def table_per_workload_winner(rows: list[Row], out_dir: Path) -> Path:
    lines: list[str] = []
    lines.append("\\begin{tabular}{llrr}")
    lines.append("\\toprule")
    lines.append("Workload & Best variant & OOMs & vs fixed\\_dual\\_mr05 \\\\")
    lines.append("\\midrule")
    for workload in _WORKLOAD_ORDER:
        best_variant: str = _HEADLINE_VARIANTS[0]
        best_oom: float = float("inf")
        for variant in _HEADLINE_VARIANTS:
            per = _sum_oom_per_workload(rows, variant)
            if per[workload] < best_oom:
                best_oom = per[workload]
                best_variant = variant
        fd05_oom = _sum_oom_per_workload(rows, "fixed_dual_mr05")[workload]
        delta = best_oom - fd05_oom
        lines.append(
            f"{_tex_escape(workload)} & {_tex_escape(best_variant)} & "
            f"{best_oom:.1f} & {delta:+.1f} \\\\"
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
    lines: list[str] = []
    lines.append("\\begin{tabular}{rrrrr}")
    lines.append("\\toprule")
    lines.append("batch\\_size & uniform\\_short & mixed\\_long & agentic\\_burst & Total \\\\")
    lines.append("\\midrule")
    for bs in _BATCH_SIZE_SERIES:
        per = _sum_oom_per_workload(rows, f"avmp_dynamic_b{bs}")
        total = sum(per.values())
        lines.append(
            f"{bs} & {per['uniform_short']:.1f} & {per['mixed_long']:.1f} & "
            f"{per['agentic_burst']:.1f} & {total:.1f} \\\\"
        )
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
        "Variant & threshold\\_low & threshold\\_high & total\\_oom & rebalance\\_count \\\\"
    )
    lines.append("\\midrule")
    for variant in _THRESHOLD_TABLE_VARIANTS:
        source = batchsize_rows if variant == "avmp_dynamic_b128" else threshold_rows
        per = _sum_oom_per_workload(source, variant)
        total = sum(per.values())
        rebal = _sum_rebalance(source, variant)
        th_low, th_high = _representative_thresholds(source, variant)
        lines.append(
            f"{_tex_escape(variant)} & {th_low:.2f} & {th_high:.2f} & "
            f"{total:.1f} & {rebal:.0f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return _write_table(out_dir, "table_stage2_threshold", "\n".join(lines) + "\n")


def generate_all(out_dir: Path) -> list[Path]:
    batchsize_rows = _load_rows(BATCHSIZE_SWEEP)
    threshold_rows = _load_rows(THRESHOLD_SWEEP)
    return [
        table_baseline_comparison(batchsize_rows, out_dir),
        table_per_workload_winner(batchsize_rows, out_dir),
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
