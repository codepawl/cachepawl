"""Paired bootstrap 95% confidence intervals for V1 headline claims.

Loads per-seed BenchmarkRun JSONs from
``benchmarks/results/avmp-v2-throughput/full/runs/<variant>/<workload>/``
and computes paired bootstrap intervals over the matched
``(workload, model, pool_bytes, seed)`` grid. Variants share random seeds
across cells, so a paired comparison is well defined: for each tuple,
the delta or ratio of variant A vs variant B is computed, and the
sampling distribution of the per-tuple summary is constructed by
resampling tuples with replacement.

Output:
- ``research/avmp/tables/generated/table_bootstrap_ci.tex`` (LaTeX tabular)
- ``research/avmp/tables/generated/bootstrap_ci.json`` (machine-readable)

The script is invoked from the project Makefile and from the paper build.
Bootstrap RNG seed is fixed at ``20260520`` for byte-identical reruns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

_FArr = npt.NDArray[np.float64]

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
RUNS_ROOT: Path = REPO_ROOT / "benchmarks/results/avmp-v2-throughput/full/runs"
OUT_TABLE: Path = REPO_ROOT / "research/avmp/tables/generated/table_bootstrap_ci.tex"
OUT_JSON: Path = REPO_ROOT / "research/avmp/tables/generated/bootstrap_ci.json"

BOOTSTRAP_ITERATIONS: int = 10000
BOOTSTRAP_SEED: int = 20260520

_FILENAME_RE = re.compile(r"^(?P<model>[^_].+?)__tb(?P<pool>\d+)gib__seed(?P<seed>\d+)\.json$")
_WORKLOADS: tuple[str, ...] = ("uniform_short", "mixed_long", "agentic_burst")
_HEADLINE_VARIANTS: tuple[str, ...] = (
    "padded_unified",
    "fixed_dual_mr05",
    "fixed_dual_mr09",
    "avmp_static_mr05",
    "avmp_dynamic_b128",
)


@dataclass(frozen=True)
class CellKey:
    workload: str
    model: str
    pool_gib: int
    seed: int


@dataclass(frozen=True)
class CellMetrics:
    oom_count: int
    goodput: float
    batch_p50: float


@dataclass(frozen=True)
class CIResult:
    label: str
    workload: str
    n_pairs: int
    point: float
    ci_low: float
    ci_high: float
    significant: bool
    statistic: Literal["delta", "ratio"]


def _parse_run_filename(path: Path) -> tuple[str, int, int] | None:
    m = _FILENAME_RE.match(path.name)
    if m is None:
        return None
    return m.group("model"), int(m.group("pool")), int(m.group("seed"))


def _load_cell(path: Path) -> CellMetrics:
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
    return CellMetrics(oom_count=oom, goodput=float(goodput), batch_p50=float(batch))


def _load_all() -> dict[str, dict[CellKey, CellMetrics]]:
    """Return ``variant -> {CellKey -> CellMetrics}``."""
    if not RUNS_ROOT.is_dir():
        raise FileNotFoundError(f"runs root missing: {RUNS_ROOT}")
    out: dict[str, dict[CellKey, CellMetrics]] = {}
    for variant in _HEADLINE_VARIANTS:
        vdir = RUNS_ROOT / variant
        if not vdir.is_dir():
            raise FileNotFoundError(f"variant runs dir missing: {vdir}")
        per_cell: dict[CellKey, CellMetrics] = {}
        for workload in _WORKLOADS:
            wdir = vdir / workload
            if not wdir.is_dir():
                raise FileNotFoundError(f"workload dir missing: {wdir}")
            for jf in sorted(wdir.glob("*.json")):
                parsed = _parse_run_filename(jf)
                if parsed is None:
                    raise ValueError(f"unparseable run filename: {jf.name}")
                model, pool_gib, seed = parsed
                key = CellKey(workload=workload, model=model, pool_gib=pool_gib, seed=seed)
                if key in per_cell:
                    raise ValueError(f"duplicate cell {key} for variant {variant}")
                per_cell[key] = _load_cell(jf)
        out[variant] = per_cell
    return out


def _paired_bootstrap_delta(
    deltas: _FArr,
    rng: np.random.Generator,
    iterations: int,
    null_value: float,
) -> tuple[float, float, float, bool]:
    """Bootstrap mean-of-deltas. Point estimate = mean(a - b)."""
    if deltas.size == 0:
        raise ValueError("no paired observations")
    n = deltas.size
    point = float(np.mean(deltas))
    idx = rng.integers(0, n, size=(iterations, n))
    boot_means = deltas[idx].mean(axis=1)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    significant = (ci_low > null_value) or (ci_high < null_value)
    return point, ci_low, ci_high, significant


def _paired_bootstrap_ratio_of_means(
    a: _FArr,
    b: _FArr,
    rng: np.random.Generator,
    iterations: int,
    null_value: float,
) -> tuple[float, float, float, bool]:
    """Bootstrap ratio-of-means. Point estimate = mean(a) / mean(b).

    Matches how the paper reports per-workload goodput ratios (eg
    "434.24 / 32.65 = 13.30x"). Each bootstrap iteration resamples
    matched ``(a_i, b_i)`` pairs with replacement and recomputes the
    ratio of the resampled means.
    """
    if a.size == 0 or a.size != b.size:
        raise ValueError("paired arrays must be same nonzero size")
    n = a.size
    eps = 1e-9
    point = float(np.mean(a) / max(float(np.mean(b)), eps))
    idx = rng.integers(0, n, size=(iterations, n))
    a_means = a[idx].mean(axis=1)
    b_means = np.maximum(b[idx].mean(axis=1), eps)
    boot = a_means / b_means
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    significant = (ci_low > null_value) or (ci_high < null_value)
    return point, ci_low, ci_high, significant


def _matched_pairs(
    data: dict[str, dict[CellKey, CellMetrics]],
    variant_a: str,
    variant_b: str,
    workload: str | None,
    field: Literal["oom_count", "goodput", "batch_p50"],
) -> tuple[_FArr, _FArr]:
    """Return aligned arrays ``(a, b)`` of the requested field for matched cells."""
    a_map = data[variant_a]
    b_map = data[variant_b]
    a_vals: list[float] = []
    b_vals: list[float] = []
    for key in sorted(
        a_map.keys(),
        key=lambda k: (k.workload, k.model, k.pool_gib, k.seed),
    ):
        if workload is not None and key.workload != workload:
            continue
        if key not in b_map:
            raise KeyError(f"missing matched cell {key} in {variant_b}")
        am = a_map[key]
        bm = b_map[key]
        a_vals.append(float(getattr(am, field)))
        b_vals.append(float(getattr(bm, field)))
    if not a_vals:
        raise ValueError(f"no matched pairs for workload={workload}")
    return np.asarray(a_vals, dtype=float), np.asarray(b_vals, dtype=float)


def _ci_delta(
    data: dict[str, dict[CellKey, CellMetrics]],
    variant_a: str,
    variant_b: str,
    workload: str | None,
    field: Literal["oom_count", "goodput", "batch_p50"],
    rng: np.random.Generator,
    label: str,
) -> CIResult:
    a, b = _matched_pairs(data, variant_a, variant_b, workload, field)
    deltas = a - b
    point, lo, hi, sig = _paired_bootstrap_delta(deltas, rng, BOOTSTRAP_ITERATIONS, null_value=0.0)
    return CIResult(
        label=label,
        workload=workload if workload is not None else "cross_workload",
        n_pairs=int(deltas.size),
        point=point,
        ci_low=lo,
        ci_high=hi,
        significant=sig,
        statistic="delta",
    )


def _ci_ratio(
    data: dict[str, dict[CellKey, CellMetrics]],
    variant_a: str,
    variant_b: str,
    workload: str | None,
    field: Literal["oom_count", "goodput", "batch_p50"],
    rng: np.random.Generator,
    label: str,
) -> CIResult:
    a, b = _matched_pairs(data, variant_a, variant_b, workload, field)
    point, lo, hi, sig = _paired_bootstrap_ratio_of_means(
        a, b, rng, BOOTSTRAP_ITERATIONS, null_value=1.0
    )
    return CIResult(
        label=label,
        workload=workload if workload is not None else "cross_workload",
        n_pairs=int(a.size),
        point=point,
        ci_low=lo,
        ci_high=hi,
        significant=sig,
        statistic="ratio",
    )


def compute_all_cis() -> list[CIResult]:
    data = _load_all()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results: list[CIResult] = []

    # Claim 1: AVMP dynamic vs best static baseline, OOM reduction
    for workload in (*_WORKLOADS, None):
        results.append(
            _ci_delta(
                data,
                "avmp_dynamic_b128",
                "fixed_dual_mr05",
                workload,
                "oom_count",
                rng,
                "avmp_dynamic_b128 - fixed_dual_mr05 (OOM count)",
            )
        )

    # Claim 2: AVMP dynamic vs best static baseline, goodput ratio
    for workload in _WORKLOADS:
        results.append(
            _ci_ratio(
                data,
                "avmp_dynamic_b128",
                "fixed_dual_mr05",
                workload,
                "goodput",
                rng,
                "avmp_dynamic_b128 / fixed_dual_mr05 (goodput ratio)",
            )
        )

    # Claim 3: AVMP static virtual handle abstraction is zero-overhead vs fixed_dual_mr05
    results.append(
        _ci_delta(
            data,
            "avmp_static_mr05",
            "fixed_dual_mr05",
            None,
            "oom_count",
            rng,
            "avmp_static_mr05 - fixed_dual_mr05 (OOM count, equivalence)",
        )
    )

    # Claim 4: SGLang default (0.9 ratio) under-performs the 0.5 ratio
    results.append(
        _ci_delta(
            data,
            "fixed_dual_mr09",
            "fixed_dual_mr05",
            None,
            "oom_count",
            rng,
            "fixed_dual_mr09 - fixed_dual_mr05 (OOM count)",
        )
    )

    # Claim 5: padded_unified is the worst baseline
    results.append(
        _ci_delta(
            data,
            "padded_unified",
            "fixed_dual_mr05",
            None,
            "oom_count",
            rng,
            "padded_unified - fixed_dual_mr05 (OOM count)",
        )
    )

    # Claim 6: effective batch sizes are identical across variants
    # (paper claim line 55: "strictly identical across all 5 variants per workload")
    for workload in _WORKLOADS:
        results.append(
            _ci_delta(
                data,
                "avmp_dynamic_b128",
                "fixed_dual_mr05",
                workload,
                "batch_p50",
                rng,
                "avmp_dynamic_b128 - fixed_dual_mr05 (effective_batch_size_p50)",
            )
        )

    return results


def _fmt(v: float, statistic: Literal["delta", "ratio"]) -> str:
    if statistic == "ratio":
        return f"{v:.2f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _escape(s: str) -> str:
    return s.replace("_", r"\_")


def write_table(results: list[CIResult]) -> None:
    """Emit ``table_bootstrap_ci.tex`` as a self-contained tabular block."""
    lines: list[str] = []
    lines.append(r"\begin{tabular}{llrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Comparison & Workload & $n$ & Point & 95\% CI & Significant \\")
    lines.append(r"\midrule")
    for r in results:
        ci = f"[{_fmt(r.ci_low, r.statistic)}, {_fmt(r.ci_high, r.statistic)}]"
        sig = r"\textbf{yes}" if r.significant else "no"
        lines.append(
            f"{_escape(r.label)} & {_escape(r.workload)} & "
            f"{r.n_pairs} & {_fmt(r.point, r.statistic)} & {ci} & {sig} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text("\n".join(lines) + "\n")


def write_json(results: list[CIResult]) -> None:
    payload = {
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "results": [
            {
                "label": r.label,
                "workload": r.workload,
                "n_pairs": r.n_pairs,
                "statistic": r.statistic,
                "point": r.point,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "significant": r.significant,
            }
            for r in results
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    results = compute_all_cis()
    write_table(results)
    write_json(results)
    print(f"Wrote {OUT_TABLE.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    for r in results:
        ci = f"[{_fmt(r.ci_low, r.statistic)}, {_fmt(r.ci_high, r.statistic)}]"
        sig = "SIG" if r.significant else "ns"
        print(
            f"  {r.label:<70s} | {r.workload:<16s} | n={r.n_pairs:>2d} "
            f"| point={_fmt(r.point, r.statistic):>7s} | CI={ci:<18s} | {sig}"
        )


if __name__ == "__main__":
    main()
