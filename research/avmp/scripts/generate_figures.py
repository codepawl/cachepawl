"""Generate paper figures from committed sweep aggregated.json files.

Four figures, each saved as a vector PDF (for the paper) and a 300 dpi
PNG (for review). All numbers come from sweep JSONs in
``benchmarks/results/``; no hardcoded values.

Headless rendering via the matplotlib Agg backend matches the
convention in :mod:`cachepawl.benchmarks.compare.plots`. PNG metadata
is stripped so reruns at the same matplotlib version are byte-identical.

Figures:

1. ``fig_oom_vs_batch_size``: line plot of mean OOMs per workload as
   ``migration_batch_size`` sweeps 1..256. Headline of stage 1.
2. ``fig_oom_comparison_final``: grouped bar chart of mean OOMs across
   four variants (padded_unified, fixed_dual_mr05, avmp_static_mr05,
   avmp_dynamic_b128) and three workloads. The paper's headline result.
3. ``fig_peak_reserved_tradeoff``: bar chart of peak_reserved_MiB,
   surfacing the 2x physical-footprint cost of AVMP honestly.
4. ``fig_threshold_sensitivity``: bar chart of cross-workload total OOMs
   across the four threshold variants + the stage 1 b128 reference.
   All five bars are at ~510 - visual proof of the stage 2 null result.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
BATCHSIZE_SWEEP: Path = REPO_ROOT / "benchmarks/results/avmp-v2-batchsize-sweep/aggregated.json"
THRESHOLD_SWEEP: Path = REPO_ROOT / "benchmarks/results/avmp-v2-threshold-sweep/aggregated.json"

_FIG_DPI: int = 300
_SAVEFIG_METADATA: dict[str, str | None] = {
    "Software": None,
    "Creator": None,
    "Date": None,
}

# ColorBrewer Set2, colorblind-safe. Order matches the four headline variants.
_PALETTE: dict[str, str] = {
    "padded_unified": "#66c2a5",
    "fixed_dual_mr05": "#fc8d62",
    "avmp_static_mr05": "#8da0cb",
    "avmp_dynamic_b128": "#e78ac3",
    # Threshold variants reuse a softer hue from the same family.
    "avmp_dynamic_b128_th_high_010": "#a6d854",
    "avmp_dynamic_b128_th_high_020": "#ffd92f",
    "avmp_dynamic_b128_th_low_002": "#e5c494",
    "avmp_dynamic_b128_th_low_010": "#b3b3b3",
}

_VARIANT_DISPLAY: dict[str, str] = {
    "padded_unified": "padded_unified",
    "fixed_dual_mr05": "fixed_dual (mr=0.5)",
    "avmp_static_mr05": "AVMP static (mr=0.5)",
    "avmp_dynamic_b128": "AVMP dynamic (b=128)",
    "avmp_dynamic_b128_th_high_010": "th_high=0.10",
    "avmp_dynamic_b128_th_high_020": "th_high=0.20",
    "avmp_dynamic_b128_th_low_002": "th_low=0.02",
    "avmp_dynamic_b128_th_low_010": "th_low=0.10",
}

_WORKLOAD_ORDER: tuple[str, ...] = ("uniform_short", "mixed_long", "agentic_burst")

_HEADLINE_VARIANTS: tuple[str, ...] = (
    "padded_unified",
    "fixed_dual_mr05",
    "avmp_static_mr05",
    "avmp_dynamic_b128",
)

_THRESHOLD_VARIANTS: tuple[str, ...] = (
    "avmp_dynamic_b128",
    "avmp_dynamic_b128_th_high_010",
    "avmp_dynamic_b128_th_high_020",
    "avmp_dynamic_b128_th_low_002",
    "avmp_dynamic_b128_th_low_010",
)

_BATCH_SIZE_SERIES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class Row:
    """Minimal row projection for figure generation.

    Mirrors the relevant fields of
    :class:`cachepawl.benchmarks.compare.aggregate.AggregatedRow` but
    avoids the dependency on that package's tuple-of-tuples
    serialization for ``allocator_specific_median``.
    """

    variant_label: str
    workload_name: str
    model_spec_name: str
    total_bytes: int
    oom_count_mean: float
    fragmentation_during_load_mean: float
    peak_reserved_bytes_mean: float
    rebalance_count_median: float


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
        fragmentation_during_load_mean=float(cast(float, raw["fragmentation_during_load_mean"])),
        peak_reserved_bytes_mean=float(cast(float, raw["peak_reserved_bytes_mean"])),
        rebalance_count_median=float(asp.get("rebalance_count", 0.0)),
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


def _save_pair(fig: Figure, out_dir: Path, basename: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{basename}.pdf"
    png_path = out_dir / f"{basename}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", metadata=_SAVEFIG_METADATA)
    fig.savefig(
        png_path,
        format="png",
        dpi=_FIG_DPI,
        bbox_inches="tight",
        metadata=_SAVEFIG_METADATA,
    )
    plt.close(fig)
    return pdf_path, png_path


def _sum_oom_per_workload(rows: list[Row], variant: str) -> dict[str, float]:
    out: dict[str, float] = {w: 0.0 for w in _WORKLOAD_ORDER}
    for r in rows:
        if r.variant_label == variant and r.workload_name in out:
            out[r.workload_name] += r.oom_count_mean
    return out


def fig_oom_vs_batch_size(rows: list[Row], out_dir: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fd05 = _sum_oom_per_workload(rows, "fixed_dual_mr05")
    for workload in _WORKLOAD_ORDER:
        ys: list[float] = []
        for bs in _BATCH_SIZE_SERIES:
            variant = f"avmp_dynamic_b{bs}"
            ys.append(_sum_oom_per_workload(rows, variant).get(workload, 0.0))
        ax.plot(
            _BATCH_SIZE_SERIES,
            ys,
            marker="o",
            label=workload.replace("_", " "),
            linewidth=1.6,
        )
        ax.axhline(
            fd05[workload],
            linestyle=":",
            color=ax.lines[-1].get_color(),
            alpha=0.55,
            linewidth=0.9,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(_BATCH_SIZE_SERIES)
    ax.set_xticklabels([str(bs) for bs in _BATCH_SIZE_SERIES])
    ax.set_xlabel("migration_batch_size")
    ax.set_ylabel("Sum of mean OOMs across 12 cells")
    ax.set_title("Stage 1: OOMs vs migration_batch_size (per workload)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.85, fontsize=8)
    return _save_pair(fig, out_dir, "fig_oom_vs_batch_size")


def fig_oom_comparison_final(rows: list[Row], out_dir: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(_WORKLOAD_ORDER), dtype=np.float64)
    width = 0.20
    offset = -1.5 * width
    for variant in _HEADLINE_VARIANTS:
        per = _sum_oom_per_workload(rows, variant)
        ys = [per[w] for w in _WORKLOAD_ORDER]
        ax.bar(
            x + offset,
            ys,
            width,
            label=_VARIANT_DISPLAY[variant],
            color=_PALETTE[variant],
            edgecolor="black",
            linewidth=0.4,
        )
        offset += width
    ax.set_xticks(x)
    ax.set_xticklabels([w.replace("_", " ") for w in _WORKLOAD_ORDER])
    ax.set_ylabel("Sum of mean OOMs across 12 cells")
    ax.set_title("Headline: cross-allocator OOM comparison (4 variants, 3 workloads)")
    ax.legend(loc="best", framealpha=0.85, fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    return _save_pair(fig, out_dir, "fig_oom_comparison_final")


def fig_peak_reserved_tradeoff(rows: list[Row], out_dir: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    mib = 1024.0 * 1024.0
    means: dict[str, float] = {}
    for variant in _HEADLINE_VARIANTS:
        peaks = [r.peak_reserved_bytes_mean for r in rows if r.variant_label == variant]
        means[variant] = (sum(peaks) / len(peaks) / mib) if peaks else 0.0
    labels = [_VARIANT_DISPLAY[v] for v in _HEADLINE_VARIANTS]
    values = [means[v] for v in _HEADLINE_VARIANTS]
    colors = [_PALETTE[v] for v in _HEADLINE_VARIANTS]
    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Mean peak_reserved (MiB)")
    ax.set_title("Peak reserved memory: AVMP carries a 2x physical-footprint cost")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ymax = max(values) * 1.15 if values else 1.0
    ax.set_ylim(0, ymax)
    for i, v in enumerate(values):
        ax.text(i, v + ymax * 0.01, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    return _save_pair(fig, out_dir, "fig_peak_reserved_tradeoff")


def fig_threshold_sensitivity(
    threshold_rows: list[Row], batchsize_rows: list[Row], out_dir: Path
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    totals: dict[str, float] = {}
    for variant in _THRESHOLD_VARIANTS:
        source = batchsize_rows if variant == "avmp_dynamic_b128" else threshold_rows
        per = _sum_oom_per_workload(source, variant)
        totals[variant] = sum(per.values())
    labels = [_VARIANT_DISPLAY.get(v, v) for v in _THRESHOLD_VARIANTS]
    values = [totals[v] for v in _THRESHOLD_VARIANTS]
    colors = [_PALETTE.get(v, "#cccccc") for v in _THRESHOLD_VARIANTS]
    positions = np.arange(len(labels), dtype=np.float64)
    ax.bar(positions, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Cross-workload total OOMs (12 cells x 3 workloads)")
    ax.set_title("Stage 2: threshold variants all tie with stage 1 b128 reference")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ymax = max(values) * 1.15 if values else 1.0
    ax.set_ylim(0, ymax)
    for i, v in enumerate(values):
        ax.text(i, v + ymax * 0.01, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    return _save_pair(fig, out_dir, "fig_threshold_sensitivity")


def generate_all(out_dir: Path) -> list[Path]:
    """Generate all four figures. Returns the list of written file paths."""

    batchsize_rows = _load_rows(BATCHSIZE_SWEEP)
    threshold_rows = _load_rows(THRESHOLD_SWEEP)
    produced: list[Path] = []
    pdf, png = fig_oom_vs_batch_size(batchsize_rows, out_dir)
    produced.extend([pdf, png])
    pdf, png = fig_oom_comparison_final(batchsize_rows, out_dir)
    produced.extend([pdf, png])
    pdf, png = fig_peak_reserved_tradeoff(batchsize_rows, out_dir)
    produced.extend([pdf, png])
    pdf, png = fig_threshold_sensitivity(threshold_rows, batchsize_rows, out_dir)
    produced.extend([pdf, png])
    return produced


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out_dir = Path(args[0]) if args else REPO_ROOT / "research/avmp/figures/generated"
    paths = generate_all(out_dir)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
