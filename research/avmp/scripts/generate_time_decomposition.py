"""Generate fig_time_decomposition stacked bars from the V1.5 sweep.

Reads ``benchmarks/results/avmp-v15-timedecomp/aggregated.json`` (or a
caller-provided path) and emits a stacked horizontal bar chart of
per-variant wall-clock decomposition: service (excl. migration),
OOM retry, migration, idle. One row per (variant, workload), bars are
normalized to fractions of cell wall time.

The figure is the empirical answer to peer review concern #3
("'faster recovery' is asserted, not measured"). If OOM retry time
dominates wall time for static baselines but not for AVMP, the
mechanism is confirmed. Otherwise the paper §4.3 mechanism claim
must be softened or revised.
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

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_INPUT: Path = REPO_ROOT / "benchmarks/results/avmp-v15-timedecomp/aggregated.json"
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "research/avmp/figures/generated"

_VARIANT_ORDER: tuple[str, ...] = (
    "padded_unified",
    "fixed_dual_mr09",
    "fixed_dual_mr05",
    "avmp_static_mr05",
    "avmp_dynamic_b128",
)
_WORKLOAD_ORDER: tuple[str, ...] = ("uniform_short", "mixed_long", "agentic_burst")

_PHASE_COLORS: dict[str, str] = {
    "service_excl_migration": "#4daf4a",
    "oom_retry": "#e41a1c",
    "migration": "#377eb8",
    "idle": "#999999",
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


def _aggregate_phase_rows(rows: Sequence[object]) -> list[_PhaseRow]:
    """Median phase timings per (variant, workload), averaged over models/pools."""
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


def _save_dual_format(fig: matplotlib.figure.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{name}.pdf"
    png_path = output_dir / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(png_path, bbox_inches="tight", dpi=150, metadata={"Creation Time": None})


def make_figure(rows: list[_PhaseRow], output_dir: Path) -> None:
    if not rows:
        raise ValueError("no phase rows to plot; sweep aggregated.json may be empty")

    labels = [f"{r.variant} / {r.workload}" for r in rows]
    fractions: dict[str, list[float]] = {k: [] for k in _PHASE_COLORS}
    for r in rows:
        wall = r.wall_ns()
        if wall == 0:
            for k in fractions:
                fractions[k].append(0.0)
            continue
        fractions["service_excl_migration"].append(r.service_excl_migration_ns / wall)
        fractions["oom_retry"].append(r.oom_retry_ns / wall)
        fractions["migration"].append(r.migration_ns / wall)
        fractions["idle"].append(r.idle_ns / wall)

    fig_h = max(4.0, 0.32 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    y = list(range(len(rows)))
    left = [0.0] * len(rows)
    for phase in ("service_excl_migration", "oom_retry", "migration", "idle"):
        vals = fractions[phase]
        ax.barh(
            y,
            vals,
            left=left,
            color=_PHASE_COLORS[phase],
            label=_PHASE_LABELS[phase],
            edgecolor="white",
            linewidth=0.4,
        )
        left = [a + b for a, b in zip(left, vals, strict=True)]

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Wall-clock fraction")
    ax.set_title("Wall-clock phase decomposition (median across model/pool cells)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save_dual_format(fig, output_dir, "fig_time_decomposition")
    plt.close(fig)


def main(input_path: Path | None = None, output_dir: Path | None = None) -> None:
    input_path = input_path or DEFAULT_INPUT
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    if not input_path.exists():
        raise FileNotFoundError(
            f"aggregated.json not found: {input_path}. Run the V1.5 time-decomposition sweep first."
        )
    rows = _aggregate_phase_rows(_load_rows(input_path))
    make_figure(rows, output_dir)
    print(f"Wrote {output_dir.relative_to(REPO_ROOT)}/fig_time_decomposition.pdf")

    # Print a quick text summary so the mechanism check is visible in the
    # PHASE4 log without having to open the figure.
    print("\nPhase decomposition (median fraction of wall time):")
    print(f"{'variant':<20s} {'workload':<16s} {'svc':>6s} {'oom':>6s} {'mig':>6s} {'idle':>6s}")
    for r in rows:
        wall = max(r.wall_ns(), 1)
        s = r.service_excl_migration_ns / wall
        o = r.oom_retry_ns / wall
        m = r.migration_ns / wall
        i = r.idle_ns / wall
        print(f"{r.variant:<20s} {r.workload:<16s} {s:>6.2%} {o:>6.2%} {m:>6.2%} {i:>6.2%}")


if __name__ == "__main__":
    main()
