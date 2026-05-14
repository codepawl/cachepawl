"""Matplotlib plots for the comparison sweep.

Both plots are saved as PNG via the Agg backend (no display required).
The savefig metadata kwargs strip the matplotlib version stamp and
creation-time fields so reruns at the same matplotlib version produce
byte-identical PNGs. Across matplotlib versions byte identity is not
guaranteed (font cache, compression table) and the reproducibility test
uses a PIL pixel diff with tolerance.

Two figures:

* ``plot_fragmentation_vs_workload``: grouped bar chart of
  ``fragmentation_final_mean`` per workload, one bar per variant, with
  error bars taken from the population std across replicates. Filtered
  to one (model_spec, total_bytes) pair to keep the figure legible.

* ``plot_padding_waste_vs_state_size``: line plot of padding waste
  versus SSM ``d_state``. ``padded_unified`` plots its allocator-specific
  ``padding_waste_bytes`` directly; ``fixed_dual_*`` plots the sum of
  ``pool_underused_bytes_kv`` and ``pool_underused_bytes_ssm`` so the
  two allocator families share one y-axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy

from cachepawl.benchmarks.compare.aggregate import AggregatedMetrics, AggregatedRow
from cachepawl.benchmarks.compare.sweep import get_model_spec, total_bytes_human

plt.rcParams["svg.hashsalt"] = "cachepawl"

_MIB: float = 1024.0 * 1024.0

_SAVEFIG_METADATA: dict[str, str | None] = {
    "Software": None,
    "Creator": None,
    "Date": None,
}

_FIG_DPI: int = 120


def plot_fragmentation_vs_workload(
    aggregated: AggregatedMetrics,
    output_path: Path,
    *,
    model_spec_filter: str = "jamba_1_5_mini",
    total_bytes_filter: int | None = None,
    git_sha: str = "",
    run_date: str = "",
) -> None:
    """Grouped bar chart: fragmentation_ratio per workload, colored by variant.

    Filters to one (model_spec, total_bytes) pair. ``total_bytes_filter``
    defaults to the largest swept value so the figure shows the
    plenty-of-room regime where fragmentation differences are clearest.
    """

    matching = [r for r in aggregated.rows if r.model_spec_name == model_spec_filter]
    if not matching:
        raise ValueError(f"no aggregated rows matched model_spec_filter={model_spec_filter!r}")
    if total_bytes_filter is None:
        total_bytes_filter = max(r.total_bytes for r in matching)
    cells = [r for r in matching if r.total_bytes == total_bytes_filter]
    if not cells:
        raise ValueError(f"no aggregated rows matched total_bytes_filter={total_bytes_filter}")

    workloads = sorted({r.workload_name for r in cells})
    variants = sorted({r.variant_label for r in cells})

    matrix_mean = numpy.zeros((len(variants), len(workloads)), dtype=numpy.float64)
    matrix_std = numpy.zeros((len(variants), len(workloads)), dtype=numpy.float64)
    for vi, variant in enumerate(variants):
        for wi, workload in enumerate(workloads):
            row = _find_row(cells, variant=variant, workload=workload)
            if row is None:
                continue
            matrix_mean[vi, wi] = row.fragmentation_final_mean
            matrix_std[vi, wi] = row.fragmentation_final_std

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=_FIG_DPI)
    bar_width = 0.8 / max(1, len(variants))
    indices = numpy.arange(len(workloads), dtype=numpy.float64)
    for vi, variant in enumerate(variants):
        offset = (vi - (len(variants) - 1) / 2.0) * bar_width
        ax.bar(
            indices + offset,
            matrix_mean[vi, :],
            width=bar_width,
            yerr=matrix_std[vi, :],
            label=variant,
            capsize=3,
        )
    ax.set_xticks(indices)
    ax.set_xticklabels(workloads, rotation=0)
    ax.set_xlabel("workload")
    ax.set_ylabel("fragmentation_ratio (final, lower is better)")
    tb_label = total_bytes_human(total_bytes_filter)
    ax.set_title(
        f"Final fragmentation by workload  (model_spec={model_spec_filter}, total_bytes={tb_label})"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    _add_watermark(fig, git_sha, run_date)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_FIG_DPI, metadata=_SAVEFIG_METADATA)
    plt.close(fig)


def plot_padding_waste_vs_state_size(
    aggregated: AggregatedMetrics,
    output_path: Path,
    *,
    workload_filter: str = "uniform_short",
    total_bytes_filter: int | None = None,
    git_sha: str = "",
    run_date: str = "",
) -> None:
    """Line plot: per-variant padding waste vs SSM d_state.

    For padded_unified, plots ``padding_waste_bytes`` directly. For
    fixed_dual variants, plots ``kv_underused + ssm_underused`` so all
    variants share one y-axis. Filter to one (workload, total_bytes)
    pair for clarity. In quick mode there is one data point per line.
    """

    matching = [r for r in aggregated.rows if r.workload_name == workload_filter]
    if not matching:
        raise ValueError(f"no aggregated rows matched workload_filter={workload_filter!r}")
    if total_bytes_filter is None:
        total_bytes_filter = max(r.total_bytes for r in matching)
    cells = [r for r in matching if r.total_bytes == total_bytes_filter]
    if not cells:
        raise ValueError(f"no aggregated rows matched total_bytes_filter={total_bytes_filter}")

    variants = sorted({r.variant_label for r in cells})

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=_FIG_DPI)
    for variant in variants:
        variant_cells = sorted(
            (r for r in cells if r.variant_label == variant),
            key=lambda r: _d_state_for_spec(r.model_spec_name),
        )
        xs = [_d_state_for_spec(r.model_spec_name) for r in variant_cells]
        ys = [_padding_waste_mib(r) for r in variant_cells]
        ax.plot(xs, ys, marker="o", label=variant)

    ax.set_xlabel("SSM d_state")
    ax.set_ylabel("padding_waste_or_pool_underused (MiB, median across replicates)")
    tb_label = total_bytes_human(total_bytes_filter)
    ax.set_title(
        f"Padding / pool-underuse vs SSM state size  "
        f"(workload={workload_filter}, total_bytes={tb_label})"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.5)
    _add_watermark(fig, git_sha, run_date)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_FIG_DPI, metadata=_SAVEFIG_METADATA)
    plt.close(fig)


def _find_row(rows: list[AggregatedRow], *, variant: str, workload: str) -> AggregatedRow | None:
    for r in rows:
        if r.variant_label == variant and r.workload_name == workload:
            return r
    return None


def _d_state_for_spec(model_spec_name: str) -> int:
    return get_model_spec(model_spec_name).ssm_profile.d_state


def _padding_waste_mib(row: AggregatedRow) -> float:
    stats = dict(row.allocator_specific_median)
    if row.allocator_name == "padded_unified":
        return float(stats.get("padding_waste_bytes", 0.0)) / _MIB
    if row.allocator_name == "fixed_dual":
        kv = float(stats.get("pool_underused_bytes_kv", 0.0))
        ssm = float(stats.get("pool_underused_bytes_ssm", 0.0))
        return (kv + ssm) / _MIB
    return 0.0


def _add_watermark(fig: object, git_sha: str, run_date: str) -> None:
    if not git_sha and not run_date:
        return
    short_sha = git_sha[:8] if git_sha else ""
    pieces: list[str] = []
    if short_sha:
        pieces.append(f"git: {short_sha}")
    if run_date:
        pieces.append(run_date)
    text = " | ".join(pieces)
    # mypy cannot infer Figure's text method without stubs; use getattr.
    text_fn = getattr(fig, "text", None)
    if callable(text_fn):
        text_fn(0.01, 0.01, text, fontsize=7, color="gray")


__all__ = [
    "plot_fragmentation_vs_workload",
    "plot_padding_waste_vs_state_size",
]
