"""Generate fig_time_decomposition stacked bars from the V1.5 sweep.

Reads ``benchmarks/results/avmp-v15-timedecomp/aggregated.json`` (or a
caller-provided path) and emits a grouped stacked bar chart of
per-variant wall-clock decomposition: service (excl. migration),
OOM retry, migration, idle. Layout mirrors ``fig_oom_comparison_final``
in ``generate_figures.py`` (vertical grouped bars per workload, theme-
aligned palette, framealpha=1.0 legend at the bottom, y-axis grid).

The figure is the empirical answer to peer review concern #3
("'faster recovery' is asserted, not measured"). If OOM retry time
dominates wall time for static baselines but not for AVMP, the
mechanism is confirmed. Otherwise the paper §4.3 mechanism claim
must be softened or revised.

We visually emphasize ``avmp_dynamic_b128`` (the focus variant) with a
thicker black bar edge and an asterisk marker above its bars.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_INPUT: Path = REPO_ROOT / "benchmarks/results/avmp-v15-timedecomp/aggregated.json"
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "research/avmp/figures/generated"

_FIG_DPI: int = 300
_SAVEFIG_METADATA: dict[str, str | None] = {
    "Software": None,
    "Creator": None,
    "Date": None,
}

_VARIANT_ORDER: tuple[str, ...] = (
    "padded_unified",
    "fixed_dual_mr09",
    "fixed_dual_mr05",
    "avmp_static_mr05",
    "avmp_dynamic_b128",
)
_VARIANT_SHORT: dict[str, str] = {
    "padded_unified": "PU",
    "fixed_dual_mr09": "FD9",
    "fixed_dual_mr05": "FD5",
    "avmp_static_mr05": "AS",
    "avmp_dynamic_b128": "AD",
}
_FOCUS_VARIANT: str = "avmp_dynamic_b128"

_WORKLOAD_ORDER: tuple[str, ...] = ("uniform_short", "mixed_long", "agentic_burst")

# Phase palette tuned for the OOM-retry slice to read as the alarm
# signal: muted green for the bulk of healthy service time, saturated
# red for OOM retry (the bottleneck on mixed_long / agentic_burst),
# steel blue for migration, neutral grey for idle. Same family as the
# ColorBrewer Set2 used in generate_figures.py but discriminable when
# stacked.
_PHASE_ORDER: tuple[str, ...] = (
    "service_excl_migration",
    "oom_retry",
    "migration",
    "idle",
)
_PHASE_COLORS: dict[str, str] = {
    "service_excl_migration": "#9ecae1",
    "oom_retry": "#e6550d",
    "migration": "#31a354",
    "idle": "#bdbdbd",
}
_PHASE_LABELS: dict[str, str] = {
    "service_excl_migration": "Service (excl. migration)",
    "oom_retry": "OOM retry",
    "migration": "Migration",
    "idle": "Idle",
}


@dataclass(frozen=True, slots=True)
class _PhaseRow:
    variant: str
    workload: str
    service_excl_migration_ns: int
    oom_retry_ns: int
    migration_ns: int
    idle_ns: int

    def wall_ns(self) -> int:
        return self.service_excl_migration_ns + self.oom_retry_ns + self.migration_ns + self.idle_ns

    def fraction(self, phase: str) -> float:
        total = self.wall_ns()
        if total == 0:
            return 0.0
        if phase == "service_excl_migration":
            return self.service_excl_migration_ns / total
        if phase == "oom_retry":
            return self.oom_retry_ns / total
        if phase == "migration":
            return self.migration_ns / total
        if phase == "idle":
            return self.idle_ns / total
        raise ValueError(f"unknown phase: {phase!r}")


def _aggregate_phase_rows(rows: Sequence[object]) -> list[_PhaseRow]:
    """Mean phase timings per (variant, workload), averaged over model/pool cells."""
    by_cell: dict[tuple[str, str], list[tuple[int, int, int, int]]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise TypeError(f"row is not a dict: {type(raw).__name__}")
        variant = str(raw.get("variant_label", ""))
        workload = str(raw.get("workload_name", ""))
        service = int(cast(int, raw.get("time_in_service_ns_median", 0)))
        oom_retry = int(cast(int, raw.get("time_in_oom_retry_ns_median", 0)))
        migration = int(cast(int, raw.get("time_in_migration_ns_median", 0)))
        idle = int(cast(int, raw.get("time_in_idle_ns_median", 0)))
        by_cell.setdefault((variant, workload), []).append((service, oom_retry, migration, idle))

    out: list[_PhaseRow] = []
    for variant in _VARIANT_ORDER:
        for workload in _WORKLOAD_ORDER:
            cells = by_cell.get((variant, workload))
            if not cells:
                continue
            mean_service = sum(c[0] for c in cells) // len(cells)
            mean_oom_retry = sum(c[1] for c in cells) // len(cells)
            mean_migration = sum(c[2] for c in cells) // len(cells)
            mean_idle = sum(c[3] for c in cells) // len(cells)
            # service includes migration; subtract for stacked display
            service_excl = max(0, mean_service - mean_migration)
            out.append(
                _PhaseRow(
                    variant=variant,
                    workload=workload,
                    service_excl_migration_ns=service_excl,
                    oom_retry_ns=mean_oom_retry,
                    migration_ns=mean_migration,
                    idle_ns=mean_idle,
                )
            )
    return out


def _load_rows(path: Path) -> list[object]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top-level must be dict")
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{path}: 'rows' must be a list")
    return list(rows)


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


def make_figure(rows: list[_PhaseRow], output_dir: Path) -> tuple[Path, Path]:
    if not rows:
        raise ValueError("no phase rows to plot; sweep aggregated.json may be empty")

    by_key: dict[tuple[str, str], _PhaseRow] = {(r.variant, r.workload): r for r in rows}

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    n_workloads = len(_WORKLOAD_ORDER)
    n_variants = len(_VARIANT_ORDER)
    x = np.arange(n_workloads, dtype=np.float64)
    width = 0.16
    half_span = (n_variants - 1) / 2.0
    offsets = np.array([(i - half_span) * width for i in range(n_variants)], dtype=np.float64)

    legend_added = {phase: False for phase in _PHASE_ORDER}

    for vi, variant in enumerate(_VARIANT_ORDER):
        is_focus = variant == _FOCUS_VARIANT
        edge_lw = 1.1 if is_focus else 0.4
        edge_color = "#1a1a1a" if is_focus else "black"
        for wi, workload in enumerate(_WORKLOAD_ORDER):
            data = by_key.get((variant, workload))
            if data is None:
                continue
            xpos = float(x[wi] + offsets[vi])
            bottom = 0.0
            for phase in _PHASE_ORDER:
                frac = data.fraction(phase)
                if frac <= 0:
                    bottom += frac
                    continue
                ax.bar(
                    xpos,
                    frac,
                    width,
                    bottom=bottom,
                    color=_PHASE_COLORS[phase],
                    edgecolor=edge_color,
                    linewidth=edge_lw,
                    label=_PHASE_LABELS[phase] if not legend_added[phase] else None,
                )
                legend_added[phase] = True
                bottom += frac
            if is_focus:
                ax.text(
                    xpos,
                    1.02,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#1a1a1a",
                )

    bar_positions: list[float] = []
    bar_labels: list[str] = []
    for wi in range(n_workloads):
        for vi, variant in enumerate(_VARIANT_ORDER):
            bar_positions.append(float(x[wi] + offsets[vi]))
            bar_labels.append(_VARIANT_SHORT[variant])

    # Variant abbreviations go on the major axis (one per bar); workload
    # labels are drawn manually below to avoid the matplotlib collision
    # that suppresses a minor-tick label sharing an x-coordinate with a
    # major tick.
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, fontsize=6.5)
    ax.tick_params(axis="x", which="major", length=0)
    for wi, workload in enumerate(_WORKLOAD_ORDER):
        ax.text(
            float(x[wi]),
            -0.085,
            workload.replace("_", " "),
            ha="center",
            va="top",
            fontsize=10,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_ylim(0.0, 1.08)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticklabels([f"{int(p * 100)}%" for p in np.linspace(0.0, 1.0, 6)])
    ax.set_ylabel("Wall-clock fraction")
    ax.set_title("Wall-clock phase decomposition (median across model/pool cells)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.04)

    handles, _labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    final_handles = []
    final_labels = []
    for phase in _PHASE_ORDER:
        target = _PHASE_LABELS[phase]
        if target in seen:
            continue
        for h in handles:
            if h.get_label() == target:
                final_handles.append(h)
                final_labels.append(target)
                seen.add(target)
                break
    focus_label = f"★ {_VARIANT_SHORT[_FOCUS_VARIANT]} = {_FOCUS_VARIANT.replace('_', ' ')} (focus)"
    ax.legend(
        final_handles,
        final_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        framealpha=1.0,
        fontsize=7,
        ncol=4,
        title=focus_label,
        title_fontsize=7,
    )

    return _save_pair(fig, output_dir, "fig_time_decomposition")


def main(input_path: Path | None = None, output_dir: Path | None = None) -> None:
    input_path = input_path or DEFAULT_INPUT
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    if not input_path.exists():
        raise FileNotFoundError(
            f"aggregated.json not found: {input_path}. Run the V1.5 time-decomposition sweep first."
        )
    rows = _aggregate_phase_rows(_load_rows(input_path))
    pdf_path, _png_path = make_figure(rows, output_dir)
    print(f"Wrote {pdf_path.relative_to(REPO_ROOT)}")

    # Print a quick text summary so the mechanism check is visible in the
    # PHASE4 log without having to open the figure.
    print("\nPhase decomposition (median fraction of wall time):")
    print(f"{'variant':<20s} {'workload':<16s} {'svc':>6s} {'oom':>6s} {'mig':>6s} {'idle':>6s}")
    for r in rows:
        s = r.fraction("service_excl_migration")
        o = r.fraction("oom_retry")
        m = r.fraction("migration")
        i = r.fraction("idle")
        print(f"{r.variant:<20s} {r.workload:<16s} {s:>6.2%} {o:>6.2%} {m:>6.2%} {i:>6.2%}")


if __name__ == "__main__":
    main()
