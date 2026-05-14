"""Replicate aggregation for the comparison sweep.

Each cell in a sweep runs ``seed_replicates`` times with distinct seeds.
This module collapses those replicates into one row per cell, holding
mean / std / min / max for the deterministic metrics, medians for
allocator-specific stats and latency percentiles, and the data needed
to compute relative improvements between two variants.

Determinism contract: on CPU, peak occupancy, fragmentation, OOM count,
and allocator-specific stats come from the allocator's own ``stats()``
hook and are bit-for-bit reproducible across reruns at the same seed.
Latency percentiles use ``time.perf_counter_ns`` and are NOT
reproducible. ``AggregatedMetrics.deterministic_subset()`` projects
only the reproducible fields so a downstream reproducibility test can
byte-compare two runs of the sweep.

Population standard deviation (ddof=0) is used throughout for stability
on small replicate counts. The 3-replicate default would yield
suspicious ``nan``s with sample std (ddof=1) on degenerate inputs.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy

from cachepawl.benchmarks import BenchmarkRun, compute_percentiles
from cachepawl.benchmarks.compare.sweep import (
    CELL_STEM_PATTERN,
    SweepResult,
)


@dataclass(frozen=True, slots=True)
class AggregatedRow:
    """One row of aggregated metrics for a (variant, workload, model_spec, tb) cell.

    ``fragmentation_during_load_*`` summaries are computed from samples
    taken while ``active_requests_samples`` was positive; this filters
    out the post-teardown sample the runner emits at end-of-trace,
    which would otherwise force the ratio to 1.0 (allocator empty).
    ``fragmentation_peak`` is the max of that filtered series.

    ``allocator_specific_median`` and ``..._iqr`` are tuples of (key, value)
    pairs sorted by key so the dataclass remains hashable. ``allocator_name``
    is the underlying class name (``padded_unified`` or ``fixed_dual``);
    ``variant_label`` is the sweep label that may carry kwargs in its name.
    """

    variant_label: str
    allocator_name: str
    workload_name: str
    model_spec_name: str
    total_bytes: int
    n_replicates: int

    peak_reserved_bytes_mean: float
    peak_reserved_bytes_std: float
    peak_reserved_bytes_min: int
    peak_reserved_bytes_max: int

    fragmentation_during_load_mean: float
    fragmentation_during_load_std: float
    fragmentation_during_load_min: float
    fragmentation_during_load_max: float
    fragmentation_peak: float

    oom_count_mean: float
    oom_count_std: float

    allocator_specific_median: tuple[tuple[str, float], ...]
    allocator_specific_iqr: tuple[tuple[str, float], ...]

    allocate_p50_ns_median: int
    allocate_p95_ns_median: int
    allocate_p99_ns_median: int

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_label": self.variant_label,
            "allocator_name": self.allocator_name,
            "workload_name": self.workload_name,
            "model_spec_name": self.model_spec_name,
            "total_bytes": self.total_bytes,
            "n_replicates": self.n_replicates,
            "peak_reserved_bytes_mean": self.peak_reserved_bytes_mean,
            "peak_reserved_bytes_std": self.peak_reserved_bytes_std,
            "peak_reserved_bytes_min": self.peak_reserved_bytes_min,
            "peak_reserved_bytes_max": self.peak_reserved_bytes_max,
            "fragmentation_during_load_mean": self.fragmentation_during_load_mean,
            "fragmentation_during_load_std": self.fragmentation_during_load_std,
            "fragmentation_during_load_min": self.fragmentation_during_load_min,
            "fragmentation_during_load_max": self.fragmentation_during_load_max,
            "fragmentation_peak": self.fragmentation_peak,
            "oom_count_mean": self.oom_count_mean,
            "oom_count_std": self.oom_count_std,
            "allocator_specific_median": dict(self.allocator_specific_median),
            "allocator_specific_iqr": dict(self.allocator_specific_iqr),
            "allocate_p50_ns_median": self.allocate_p50_ns_median,
            "allocate_p95_ns_median": self.allocate_p95_ns_median,
            "allocate_p99_ns_median": self.allocate_p99_ns_median,
        }

    def deterministic_subset(self) -> dict[str, object]:
        """Project only fields that are bit-stable across reruns on CPU."""

        return {
            "variant_label": self.variant_label,
            "allocator_name": self.allocator_name,
            "workload_name": self.workload_name,
            "model_spec_name": self.model_spec_name,
            "total_bytes": self.total_bytes,
            "n_replicates": self.n_replicates,
            "peak_reserved_bytes_mean": self.peak_reserved_bytes_mean,
            "peak_reserved_bytes_std": self.peak_reserved_bytes_std,
            "peak_reserved_bytes_min": self.peak_reserved_bytes_min,
            "peak_reserved_bytes_max": self.peak_reserved_bytes_max,
            "fragmentation_during_load_mean": self.fragmentation_during_load_mean,
            "fragmentation_during_load_std": self.fragmentation_during_load_std,
            "fragmentation_during_load_min": self.fragmentation_during_load_min,
            "fragmentation_during_load_max": self.fragmentation_during_load_max,
            "fragmentation_peak": self.fragmentation_peak,
            "oom_count_mean": self.oom_count_mean,
            "oom_count_std": self.oom_count_std,
            "allocator_specific_median": dict(self.allocator_specific_median),
            "allocator_specific_iqr": dict(self.allocator_specific_iqr),
        }


@dataclass(frozen=True, slots=True)
class AggregatedMetrics:
    """All aggregated rows from one sweep, in stable insertion order."""

    rows: tuple[AggregatedRow, ...]

    def to_dict(self) -> dict[str, object]:
        return {"rows": [r.to_dict() for r in self.rows]}

    def deterministic_subset(self) -> dict[str, object]:
        return {"rows": [r.deterministic_subset() for r in self.rows]}


@dataclass(frozen=True, slots=True)
class _CellKey:
    variant_label: str
    workload_name: str
    model_spec_name: str
    total_bytes: int


def aggregate_runs(result: SweepResult) -> AggregatedMetrics:
    """Aggregate replicates from a SweepResult into one row per cell.

    Cells are keyed by (variant_label, workload, model_spec, total_bytes).
    Within a key, all runs sharing that key are pooled and reduced.
    """

    label_to_allocator: dict[str, str] = {v.label: v.allocator_name for v in result.config.variants}
    groups: dict[_CellKey, list[BenchmarkRun]] = defaultdict(list)
    order: list[_CellKey] = []
    for idx, run in enumerate(result.runs):
        stem = result.cell_stems[idx]
        model_spec_name, tb_label, _seed = _parse_stem(stem)
        total_bytes = _parse_tb_label(tb_label)
        key = _CellKey(
            variant_label=run.allocator_name,
            workload_name=run.spec.name,
            model_spec_name=model_spec_name,
            total_bytes=total_bytes,
        )
        if key not in groups:
            order.append(key)
        groups[key].append(run)

    rows: list[AggregatedRow] = []
    for key in order:
        rows.append(
            _aggregate_one_cell(
                key=key,
                runs=groups[key],
                allocator_name=label_to_allocator.get(key.variant_label, key.variant_label),
            )
        )
    return AggregatedMetrics(rows=tuple(rows))


def _aggregate_one_cell(
    *,
    key: _CellKey,
    runs: Sequence[BenchmarkRun],
    allocator_name: str,
) -> AggregatedRow:
    if not runs:
        raise ValueError(f"cannot aggregate empty replicate list for {key!r}")
    peak_reserved = [r.metrics.peak_reserved_bytes for r in runs]
    per_run_during_load = [_fragmentation_during_load(r) for r in runs]
    per_run_mean_frag = [_mean(samples) for samples in per_run_during_load]
    per_run_peak_frag = [max(samples) if samples else 0.0 for samples in per_run_during_load]
    ooms = [r.metrics.oom_count for r in runs]
    p50s = [compute_percentiles(r.metrics.allocate_latency_ns).p50_ns for r in runs]
    p95s = [compute_percentiles(r.metrics.allocate_latency_ns).p95_ns for r in runs]
    p99s = [compute_percentiles(r.metrics.allocate_latency_ns).p99_ns for r in runs]

    per_run_stats = [r.metrics.allocator_specific_stats for r in runs]
    allocator_specific = _aggregate_allocator_specific(per_run_stats)

    return AggregatedRow(
        variant_label=key.variant_label,
        allocator_name=allocator_name,
        workload_name=key.workload_name,
        model_spec_name=key.model_spec_name,
        total_bytes=key.total_bytes,
        n_replicates=len(runs),
        peak_reserved_bytes_mean=_mean(peak_reserved),
        peak_reserved_bytes_std=_std(peak_reserved),
        peak_reserved_bytes_min=min(peak_reserved),
        peak_reserved_bytes_max=max(peak_reserved),
        fragmentation_during_load_mean=_mean(per_run_mean_frag),
        fragmentation_during_load_std=_std(per_run_mean_frag),
        fragmentation_during_load_min=min(per_run_mean_frag) if per_run_mean_frag else 0.0,
        fragmentation_during_load_max=max(per_run_mean_frag) if per_run_mean_frag else 0.0,
        fragmentation_peak=max(per_run_peak_frag) if per_run_peak_frag else 0.0,
        oom_count_mean=_mean(ooms),
        oom_count_std=_std(ooms),
        allocator_specific_median=allocator_specific[0],
        allocator_specific_iqr=allocator_specific[1],
        allocate_p50_ns_median=int(statistics.median(p50s)),
        allocate_p95_ns_median=int(statistics.median(p95s)),
        allocate_p99_ns_median=int(statistics.median(p99s)),
    )


def _fragmentation_during_load(run: BenchmarkRun) -> list[float]:
    """Fragmentation samples taken while at least one request was active.

    The runner emits one final ``collector.sample`` call AFTER the event
    loop drains, at which point every request has departed and
    ``allocated == 0`` forces fragmentation to ~1.0. That tail sample is
    a teardown artifact, not a measurement of pool behavior under load.
    Filter it (and any other idle ticks) by pairing each sample with
    its ``active_requests_samples`` entry and keeping only entries with
    positive active count. Empty result is legal (smoke runs that fit
    in one tick); callers fall back to 0.0 for the corresponding
    summary statistics.
    """

    frags = run.metrics.fragmentation_samples
    actives = run.metrics.active_requests_samples
    n = min(len(frags), len(actives))
    return [float(frags[i]) for i in range(n) if actives[i] > 0]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(numpy.mean(numpy.asarray(values, dtype=numpy.float64)))


def _std(values: Sequence[float]) -> float:
    """Population standard deviation (ddof=0)."""

    if not values:
        return 0.0
    return float(numpy.std(numpy.asarray(values, dtype=numpy.float64), ddof=0))


def _aggregate_allocator_specific(
    per_run: Sequence[Mapping[str, float]],
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    """Return (median, IQR) tuples for the union of allocator-specific keys.

    Keys missing from a replicate are filled with 0.0 (the runner emits
    every documented key on every run, so this only triggers when a
    test fixture omits keys deliberately). Output tuples are sorted by
    key for deterministic serialization.
    """

    if not per_run:
        return ((), ())
    all_keys: set[str] = set()
    for stats in per_run:
        all_keys.update(stats.keys())
    median_pairs: list[tuple[str, float]] = []
    iqr_pairs: list[tuple[str, float]] = []
    for k in sorted(all_keys):
        values = numpy.asarray([float(stats.get(k, 0.0)) for stats in per_run], dtype=numpy.float64)
        median_pairs.append((k, float(numpy.median(values))))
        iqr_pairs.append((k, float(numpy.percentile(values, 75) - numpy.percentile(values, 25))))
    return tuple(median_pairs), tuple(iqr_pairs)


_LOWER_IS_BETTER_DEFAULT: tuple[str, ...] = (
    "fragmentation_during_load_mean",
    "peak_reserved_bytes_mean",
    "oom_count_mean",
)


def compute_relative_improvement(
    baseline_rows: Sequence[AggregatedRow],
    target_rows: Sequence[AggregatedRow],
    *,
    lower_is_better: tuple[str, ...] = _LOWER_IS_BETTER_DEFAULT,
) -> dict[tuple[str, str, str, int], dict[str, float]]:
    """Return target-vs-baseline percent improvements per matched cell.

    Each target row is matched against the baseline row at the same
    (workload, model_spec, total_bytes). The output is keyed by
    ``(workload, model_spec, target_variant, total_bytes)`` and maps
    each metric in ``lower_is_better`` to the percent improvement:

        pct_better = (baseline - target) / baseline * 100

    A positive value means the target beat the baseline. ``baseline``
    of zero yields ``inf`` when target differs and ``0.0`` when they
    match, to avoid spurious division-by-zero crashes.
    """

    baseline_by_cell: dict[tuple[str, str, int], AggregatedRow] = {}
    for row in baseline_rows:
        cell = (row.workload_name, row.model_spec_name, row.total_bytes)
        if cell in baseline_by_cell:
            raise ValueError(
                f"baseline_rows contains duplicate cell {cell}; "
                "expected exactly one baseline variant per cell"
            )
        baseline_by_cell[cell] = row

    out: dict[tuple[str, str, str, int], dict[str, float]] = {}
    for row in target_rows:
        cell = (row.workload_name, row.model_spec_name, row.total_bytes)
        if cell not in baseline_by_cell:
            continue
        baseline = baseline_by_cell[cell]
        key = (row.workload_name, row.model_spec_name, row.variant_label, row.total_bytes)
        deltas: dict[str, float] = {}
        for metric in lower_is_better:
            b_val = _row_metric(baseline, metric)
            t_val = _row_metric(row, metric)
            deltas[metric] = _percent_better(b_val, t_val)
        out[key] = deltas
    return out


def _row_metric(row: AggregatedRow, metric: str) -> float:
    if not hasattr(row, metric):
        raise ValueError(f"AggregatedRow has no attribute {metric!r}")
    value = getattr(row, metric)
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"metric {metric!r} is not numeric on AggregatedRow (got {type(value).__name__})"
        )
    return float(value)


def _percent_better(baseline: float, target: float) -> float:
    if baseline == 0.0:
        if target == 0.0:
            return 0.0
        return float("inf") if target < 0.0 else float("-inf")
    return (baseline - target) / baseline * 100.0


_STEM_RE = re.compile(CELL_STEM_PATTERN)


def _parse_stem(stem: str) -> tuple[str, str, int]:
    match = _STEM_RE.match(stem)
    if match is None:
        raise ValueError(f"stem does not match grammar {CELL_STEM_PATTERN!r}: {stem!r}")
    return match.group("model_spec"), match.group("tb"), int(match.group("seed"))


def _parse_tb_label(label: str) -> int:
    if label.endswith("gib"):
        return int(label[:-3]) * 1024**3
    if label.endswith("mib"):
        return int(label[:-3]) * 1024**2
    if label.endswith("b"):
        return int(label[:-1])
    raise ValueError(f"unknown size label: {label!r}")


__all__ = [
    "AggregatedMetrics",
    "AggregatedRow",
    "aggregate_runs",
    "compute_relative_improvement",
]
