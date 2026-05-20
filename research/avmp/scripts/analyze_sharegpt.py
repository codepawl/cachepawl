"""ShareGPT trace replay analysis for V1.5 §4.3.5.

Reads ``benchmarks/results/avmp-v15-sharegpt/`` per-seed BenchmarkRun
JSONs and emits:

1. ``research/avmp/tables/generated/table_sharegpt_results.tex`` -
   per-variant OOM / goodput / batch_p50 with paired bootstrap CIs for
   the AVMP-vs-fixed_dual_mr05 deltas.
2. ``research/avmp/figures/generated/fig_sharegpt_vs_synthetic.{pdf,png}`` -
   grouped bars showing AVMP goodput ratio and OOM reduction on the
   ShareGPT workload alongside the V1 synthetic workloads (data sourced
   from ``benchmarks/results/avmp-v2-throughput/full/runs/``).

The script is intentionally honest about whatever the data shows. If
ShareGPT-specific numbers are lower than the synthetic averages, the
figure must show that, and the paper text in §4.3.5 / §4.5 must
acknowledge it.

Paired bootstrap follows the same conventions as ``bootstrap_ci.py``
(matched `(model, pool, seed)` tuples, B=10000, RNG seed 20260520).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SHAREGPT_RUNS: Path = REPO_ROOT / "benchmarks/results/avmp-v15-sharegpt/runs"
V1_RUNS: Path = REPO_ROOT / "benchmarks/results/avmp-v2-throughput/full/runs"
OUT_TABLE: Path = REPO_ROOT / "research/avmp/tables/generated/table_sharegpt_results.tex"
OUT_FIG_DIR: Path = REPO_ROOT / "research/avmp/figures/generated"

BOOTSTRAP_ITERATIONS: int = 10000
BOOTSTRAP_SEED: int = 20260520

_FILENAME_RE = re.compile(r"^(?P<model>[^_].+?)__tb(?P<pool>\d+)gib__seed(?P<seed>\d+)\.json$")
_HEADLINE_VARIANTS: tuple[str, ...] = (
    "padded_unified",
    "fixed_dual_mr05",
    "fixed_dual_mr09",
    "avmp_static_mr05",
    "avmp_dynamic_b128",
)
_SYNTHETIC_WORKLOADS: tuple[str, ...] = ("uniform_short", "mixed_long", "agentic_burst")

_FArr = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _CellKey:
    workload: str
    model: str
    pool_gib: int
    seed: int


@dataclass(frozen=True, slots=True)
class _CellMetrics:
    oom_count: int
    goodput: float
    batch_p50: float


def _parse_run_filename(path: Path) -> tuple[str, int, int] | None:
    m = _FILENAME_RE.match(path.name)
    if m is None:
        return None
    return m.group("model"), int(m.group("pool")), int(m.group("seed"))


def _load_cell(path: Path) -> _CellMetrics:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: top-level JSON must be an object")
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError(f"{path}: metrics must be a dict")
    oom = metrics.get("oom_count")
    goodput = metrics.get("goodput_requests_per_second")
    batch = metrics.get("effective_batch_size_p50")
    if not isinstance(oom, int):
        raise TypeError(f"{path}: oom_count must be int")
    if not isinstance(goodput, (int, float)):
        raise TypeError(f"{path}: goodput_requests_per_second must be numeric")
    if not isinstance(batch, (int, float)):
        raise TypeError(f"{path}: effective_batch_size_p50 must be numeric")
    return _CellMetrics(oom_count=oom, goodput=float(goodput), batch_p50=float(batch))


def _load_variant(
    root: Path, variant: str, workloads: Sequence[str]
) -> dict[_CellKey, _CellMetrics]:
    vdir = root / variant
    if not vdir.is_dir():
        raise FileNotFoundError(f"missing variant dir: {vdir}")
    out: dict[_CellKey, _CellMetrics] = {}
    for workload in workloads:
        wdir = vdir / workload
        if not wdir.is_dir():
            raise FileNotFoundError(f"missing workload dir: {wdir}")
        for jf in sorted(wdir.glob("*.json")):
            parsed = _parse_run_filename(jf)
            if parsed is None:
                raise ValueError(f"unparseable filename: {jf.name}")
            model, pool, seed = parsed
            key = _CellKey(workload=workload, model=model, pool_gib=pool, seed=seed)
            if key in out:
                raise ValueError(f"duplicate cell {key}")
            out[key] = _load_cell(jf)
    return out


def _paired_arrays(
    a_map: dict[_CellKey, _CellMetrics],
    b_map: dict[_CellKey, _CellMetrics],
    field: Literal["oom_count", "goodput", "batch_p50"],
    workload: str | None,
) -> tuple[_FArr, _FArr]:
    a_vals: list[float] = []
    b_vals: list[float] = []
    for key in sorted(a_map.keys(), key=lambda k: (k.workload, k.model, k.pool_gib, k.seed)):
        if workload is not None and key.workload != workload:
            continue
        if key not in b_map:
            raise KeyError(f"missing matched cell {key}")
        a_vals.append(float(getattr(a_map[key], field)))
        b_vals.append(float(getattr(b_map[key], field)))
    if not a_vals:
        raise ValueError(f"no matched pairs for workload={workload}")
    return np.asarray(a_vals, dtype=float), np.asarray(b_vals, dtype=float)


def _bootstrap_delta(
    a: _FArr, b: _FArr, rng: np.random.Generator
) -> tuple[float, float, float, bool]:
    deltas = a - b
    n = deltas.size
    point = float(np.mean(deltas))
    idx = rng.integers(0, n, size=(BOOTSTRAP_ITERATIONS, n))
    means = deltas[idx].mean(axis=1)
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    sig = lo > 0 or hi < 0
    return point, lo, hi, sig


def _bootstrap_ratio_of_means(
    a: _FArr, b: _FArr, rng: np.random.Generator
) -> tuple[float, float, float, bool]:
    n = a.size
    eps = 1e-9
    point = float(np.mean(a) / max(float(np.mean(b)), eps))
    idx = rng.integers(0, n, size=(BOOTSTRAP_ITERATIONS, n))
    a_means = a[idx].mean(axis=1)
    b_means = np.maximum(b[idx].mean(axis=1), eps)
    boot = a_means / b_means
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    sig = lo > 1.0 or hi < 1.0
    return point, lo, hi, sig


@dataclass(frozen=True, slots=True)
class _PerVariantRow:
    variant: str
    n_cells: int
    oom_mean: float
    goodput_mean: float
    batch_p50_mean: float
    oom_delta_vs_baseline: tuple[float, float, float, bool] | None
    goodput_ratio_vs_baseline: tuple[float, float, float, bool] | None


def _build_table_rows(
    sharegpt_data: dict[str, dict[_CellKey, _CellMetrics]],
    baseline_variant: str,
    rng: np.random.Generator,
) -> list[_PerVariantRow]:
    baseline_cells = sharegpt_data[baseline_variant]
    rows: list[_PerVariantRow] = []
    for variant in _HEADLINE_VARIANTS:
        if variant not in sharegpt_data:
            continue
        cells = sharegpt_data[variant]
        values = list(cells.values())
        ooms = np.asarray([v.oom_count for v in values], dtype=float)
        goodputs = np.asarray([v.goodput for v in values], dtype=float)
        batches = np.asarray([v.batch_p50 for v in values], dtype=float)
        if variant == baseline_variant:
            oom_delta = None
            gp_ratio = None
        else:
            a_oom, b_oom = _paired_arrays(cells, baseline_cells, "oom_count", None)
            a_gp, b_gp = _paired_arrays(cells, baseline_cells, "goodput", None)
            oom_delta = _bootstrap_delta(a_oom, b_oom, rng)
            gp_ratio = _bootstrap_ratio_of_means(a_gp, b_gp, rng)
        rows.append(
            _PerVariantRow(
                variant=variant,
                n_cells=len(values),
                oom_mean=float(np.mean(ooms)),
                goodput_mean=float(np.mean(goodputs)),
                batch_p50_mean=float(np.mean(batches)),
                oom_delta_vs_baseline=oom_delta,
                goodput_ratio_vs_baseline=gp_ratio,
            )
        )
    return rows


def _fmt(v: float, prec: int = 2) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.{prec}f}"


def _escape(s: str) -> str:
    return s.replace("_", r"\_")


def write_table(rows: list[_PerVariantRow], baseline_variant: str) -> None:
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(r"\begin{tabular}{lrrrll}")
    lines.append(r"\toprule")
    lines.append(
        r"Variant & $\bar{N}_{\mathrm{OOM}}$ & Goodput (req/s) & $\bar{B}_{p50}$ & "
        r"$\Delta N_{\mathrm{OOM}}$ vs " + _escape(baseline_variant) + r" (95\% CI) & "
        r"Goodput ratio (95\% CI) \\"
    )
    lines.append(r"\midrule")
    for r in rows:
        oom_cell = "--"
        gp_cell = "--"
        if r.oom_delta_vs_baseline is not None:
            point, lo, hi, sig = r.oom_delta_vs_baseline
            marker = r"\textbf{*}" if sig else ""
            oom_cell = f"{_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}] {marker}".strip()
        if r.goodput_ratio_vs_baseline is not None:
            point, lo, hi, sig = r.goodput_ratio_vs_baseline
            marker = r"\textbf{*}" if sig else ""
            gp_cell = f"{_fmt(point)}$\\times$ [{_fmt(lo)}, {_fmt(hi)}] {marker}".strip()
        lines.append(
            f"{_escape(r.variant)} & {_fmt(r.oom_mean)} & {_fmt(r.goodput_mean)} & "
            f"{_fmt(r.batch_p50_mean)} & {oom_cell} & {gp_cell} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    OUT_TABLE.write_text("\n".join(lines) + "\n")


def _save_dual_format(fig: matplotlib.figure.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(
        output_dir / f"{name}.png",
        bbox_inches="tight",
        dpi=150,
        metadata={"Creation Time": None},
    )


def write_figure(
    sharegpt_data: dict[str, dict[_CellKey, _CellMetrics]],
    v1_data: dict[str, dict[_CellKey, _CellMetrics]],
    rng: np.random.Generator,
) -> None:
    """Grouped bars: AVMP vs fixed_dual_mr05 goodput ratio per workload."""

    workloads = [*_SYNTHETIC_WORKLOADS, "sharegpt_replay"]
    ratios: list[float] = []
    ci_los: list[float] = []
    ci_his: list[float] = []
    for workload in workloads:
        if workload == "sharegpt_replay":
            avmp_cells = sharegpt_data.get("avmp_dynamic_b128", {})
            base_cells = sharegpt_data.get("fixed_dual_mr05", {})
        else:
            avmp_cells = v1_data["avmp_dynamic_b128"]
            base_cells = v1_data["fixed_dual_mr05"]
        if not avmp_cells:
            ratios.append(0.0)
            ci_los.append(0.0)
            ci_his.append(0.0)
            continue
        a, b = _paired_arrays(
            avmp_cells, base_cells, "goodput", workload if workload != "sharegpt_replay" else None
        )
        point, lo, hi, _sig = _bootstrap_ratio_of_means(a, b, rng)
        ratios.append(point)
        ci_los.append(lo)
        ci_his.append(hi)

    x = np.arange(len(workloads))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    colors = ["#377eb8", "#377eb8", "#377eb8", "#4daf4a"]
    bars = ax.bar(x, ratios, color=colors, edgecolor="white", linewidth=0.5)
    err_low = [max(0.0, r - lo) for r, lo in zip(ratios, ci_los, strict=True)]
    err_high = [hi - r for r, hi in zip(ratios, ci_his, strict=True)]
    ax.errorbar(x, ratios, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=3)
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--", label="Equality")
    ax.set_xticks(x)
    ax.set_xticklabels([w.replace("_", " ") for w in workloads], rotation=15, ha="right")
    ax.set_ylabel(r"avmp\_dynamic\_b128 / fixed\_dual\_mr05 goodput")
    ax.set_title("AVMP goodput advantage on synthetic and ShareGPT workloads")
    ymax = max(r + h for r, h in zip(ratios, err_high, strict=True)) * 1.12
    ax.set_ylim(0.0, ymax)
    for rect, point, hi_err in zip(bars, ratios, err_high, strict=True):
        ax.annotate(
            f"{point:.2f}x",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height() + hi_err),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_dual_format(fig, OUT_FIG_DIR, "fig_sharegpt_vs_synthetic")
    plt.close(fig)


def main() -> None:
    if not SHAREGPT_RUNS.is_dir():
        raise FileNotFoundError(
            f"sharegpt sweep runs not found: {SHAREGPT_RUNS}. Run the V1.5 sharegpt sweep first."
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sharegpt_data: dict[str, dict[_CellKey, _CellMetrics]] = {}
    for variant in _HEADLINE_VARIANTS:
        if not (SHAREGPT_RUNS / variant).is_dir():
            print(f"  skipping {variant}: not in sharegpt runs")
            continue
        sharegpt_data[variant] = _load_variant(SHAREGPT_RUNS, variant, ("sharegpt_replay",))

    v1_data: dict[str, dict[_CellKey, _CellMetrics]] = {}
    for variant in _HEADLINE_VARIANTS:
        v1_data[variant] = _load_variant(V1_RUNS, variant, _SYNTHETIC_WORKLOADS)

    rows = _build_table_rows(sharegpt_data, "fixed_dual_mr05", rng)
    write_table(rows, "fixed_dual_mr05")
    write_figure(sharegpt_data, v1_data, rng)

    print(f"Wrote {OUT_TABLE.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_FIG_DIR.relative_to(REPO_ROOT)}/fig_sharegpt_vs_synthetic.pdf")
    print("\nShareGPT per-variant summary:")
    print(
        f"{'variant':<22s} {'n':>3s} {'OOM mean':>9s} {'goodput':>9s} "
        f"{'batch p50':>9s} {'OOM delta CI':>22s} {'goodput ratio CI':>22s}"
    )
    for r in rows:
        oom_ci = "--"
        gp_ci = "--"
        if r.oom_delta_vs_baseline is not None:
            p, lo, hi, _s = r.oom_delta_vs_baseline
            oom_ci = f"{_fmt(p)} [{_fmt(lo)}, {_fmt(hi)}]"
        if r.goodput_ratio_vs_baseline is not None:
            p, lo, hi, _s = r.goodput_ratio_vs_baseline
            gp_ci = f"{_fmt(p)}x [{_fmt(lo)}, {_fmt(hi)}]"
        print(
            f"{r.variant:<22s} {r.n_cells:>3d} {r.oom_mean:>9.2f} {r.goodput_mean:>9.2f} "
            f"{r.batch_p50_mean:>9.2f} {oom_ci:>22s} {gp_ci:>22s}"
        )


if __name__ == "__main__":
    main()
