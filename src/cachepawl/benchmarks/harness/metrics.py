"""Metrics collection for benchmark runs.

The collector is a context manager. Before entry on CUDA it resets the
peak-memory counters and optionally enables PyTorch's allocator-history
recorder. On exit it captures the peak occupancy figures and dumps the
optional snapshot pickle for later inspection in the PyTorch Memory Viz
tool.

Latency samples are recorded in nanoseconds via ``time.perf_counter_ns``.
The runner is expected to wrap each ``allocator.allocate`` and
``allocator.free`` call and feed the measured delta into
``record_allocate`` and ``record_free``.

Fragmentation sampling is event-count driven, not wall-clock driven, so
that runs are deterministic for a given seed across machines. The
``sample_every_n_events`` knob lives on the runner side, not the
collector; the collector exposes the explicit ``sample`` entry point and
trusts the caller to invoke it at the chosen cadence.

References:
- torch.cuda.memory_allocated / memory_reserved / max_memory_*:
  https://docs.pytorch.org/docs/2.12/cuda.html
- torch.cuda.memory._record_memory_history and _dump_snapshot:
  https://pytorch.org/blog/understanding-gpu-memory-1/
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import TracebackType

import numpy as np
import torch

from cachepawl.allocator.base import Allocator


@dataclass(frozen=True, slots=True)
class LatencyPercentiles:
    """Summary statistics for a latency sample list."""

    p50_ns: int
    p95_ns: int
    p99_ns: int
    max_ns: int


@dataclass(slots=True)
class AllocatorMetrics:
    """Per-run metrics captured by ``MetricsCollector``.

    All fields are aggregate over the entire run. Per-sample lists are
    in order of capture; the runner is responsible for ordering them
    coherently.

    ``allocator_specific_stats`` is a free-form mapping that lets
    individual allocators surface their characteristic metrics
    (padding waste, per-pool underuse) without polluting the shared
    field set. Convention: keys are strings, values are ``float``.
    Counts and bytes get converted to ``float`` at record time.
    Non-numeric tags belong in ``BenchmarkRun.notes`` instead.

    Throughput fields (added in schema 1.2.0) are derived at the end
    of the run by ``MetricsCollector.finalize_throughput_metrics``.
    ``effective_batch_size_*`` are statistics of
    ``active_requests_samples`` filtered to ticks with at least one
    in-flight request. ``goodput_requests_per_second`` is total
    successfully completed requests divided by wall-clock seconds.
    ``completion_ratio`` is completed divided by submitted, where
    "completed" is strict: a request counts iff no OOM occurred
    during its lifetime AND every block it held was freed cleanly.
    ``time_to_first_oom_seconds`` is wall-clock time from the start
    of the metrics window to the first ``record_oom`` call; ``None``
    if the run had no OOMs.
    """

    peak_reserved_bytes: int = 0
    peak_allocated_bytes: int = 0
    fragmentation_samples: list[float] = field(default_factory=list)
    allocate_latency_ns: list[int] = field(default_factory=list)
    free_latency_ns: list[int] = field(default_factory=list)
    oom_count: int = 0
    preemption_count: int = 0
    active_requests_samples: list[int] = field(default_factory=list)
    effective_batch_size_mean: float = 0.0
    effective_batch_size_p50: float = 0.0
    effective_batch_size_p95: float = 0.0
    effective_batch_size_p99: float = 0.0
    goodput_requests_per_second: float = 0.0
    completion_ratio: float = 0.0
    time_to_first_oom_seconds: float | None = None
    allocator_specific_stats: dict[str, float] = field(default_factory=dict)

    def allocate_latency_percentiles(self) -> LatencyPercentiles:
        return compute_percentiles(self.allocate_latency_ns)

    def free_latency_percentiles(self) -> LatencyPercentiles:
        return compute_percentiles(self.free_latency_ns)


def compute_percentiles(samples: list[int]) -> LatencyPercentiles:
    """Return p50, p95, p99, and max for ``samples`` in nanoseconds.

    Empty input maps to all-zero percentiles so callers can serialize an
    empty run without a special case.
    """

    if not samples:
        return LatencyPercentiles(p50_ns=0, p95_ns=0, p99_ns=0, max_ns=0)
    arr = np.asarray(samples, dtype=np.int64)
    return LatencyPercentiles(
        p50_ns=int(np.percentile(arr, 50)),
        p95_ns=int(np.percentile(arr, 95)),
        p99_ns=int(np.percentile(arr, 99)),
        max_ns=int(arr.max()),
    )


class MetricsCollector:
    """Context manager that captures per-run allocator metrics.

    On CUDA it reads ``torch.cuda.memory_allocated`` and
    ``torch.cuda.memory_reserved`` at each sample tick and computes
    fragmentation as ``1 - allocated / reserved``. On CPU it falls back
    to ``allocator.stats()`` so that the harness runs in CI without a
    GPU.
    """

    def __init__(
        self,
        device: str,
        allocator: Allocator,
        record_memory_snapshot: bool = False,
        snapshot_path: str | None = None,
        snapshot_max_entries: int = 100_000,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"unsupported device {device!r}; expected 'cpu' or 'cuda'")
        if record_memory_snapshot and snapshot_path is None:
            raise ValueError("snapshot_path is required when record_memory_snapshot is True")
        self._device = device
        self._allocator = allocator
        self._record_memory_snapshot = record_memory_snapshot
        self._snapshot_path = snapshot_path
        self._snapshot_max_entries = snapshot_max_entries
        self._metrics = AllocatorMetrics()
        self._total_submitted: int = 0
        self._total_completed: int = 0
        self._first_oom_wall_ns: int | None = None
        self._run_start_wall_ns: int | None = None
        self._run_end_wall_ns: int | None = None

    def __enter__(self) -> MetricsCollector:
        if self._device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            if self._record_memory_snapshot:
                _record_memory_history_enable(self._snapshot_max_entries)
        # Wall-clock window for goodput excludes torch's reset cost; we
        # stamp the start AFTER reset_peak_memory_stats so the duration
        # reflects user-observable simulation time only.
        self._run_start_wall_ns = time.perf_counter_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Stamp the end BEFORE peak reads so goodput reflects the
        # simulation window, not collector teardown overhead.
        self._run_end_wall_ns = time.perf_counter_ns()
        if self._device == "cuda":
            self._metrics.peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
            self._metrics.peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
            if self._record_memory_snapshot and self._snapshot_path is not None:
                _record_memory_history_dump(self._snapshot_path)
        else:
            astats = self._allocator.stats()
            self._metrics.peak_allocated_bytes = astats.allocated_blocks
            self._metrics.peak_reserved_bytes = astats.total_blocks

    def record_allocate(self, latency_ns: int) -> None:
        self._metrics.allocate_latency_ns.append(latency_ns)

    def record_free(self, latency_ns: int) -> None:
        self._metrics.free_latency_ns.append(latency_ns)

    def record_oom(self) -> None:
        self._metrics.oom_count += 1
        if self._first_oom_wall_ns is None:
            self._first_oom_wall_ns = time.perf_counter_ns()

    def record_preemption(self) -> None:
        self._metrics.preemption_count += 1

    def record_submitted(self) -> None:
        """Mark that one request entered the system (arrival processed)."""

        self._total_submitted += 1

    def record_completed(self) -> None:
        """Mark that one request completed cleanly (no OOM, freed)."""

        self._total_completed += 1

    def finalize_throughput_metrics(self) -> None:
        """Populate derived throughput fields from collector state.

        Must be called AFTER ``__exit__`` (so wall-clock stamps and
        sample lists are final) and BEFORE serializing the metrics.
        Safe to call once; calling twice overwrites with the same
        values.
        """

        samples = self._metrics.active_requests_samples
        filtered = [x for x in samples if x > 0]
        if filtered:
            arr = np.asarray(filtered, dtype=np.float64)
            self._metrics.effective_batch_size_mean = float(arr.mean())
            self._metrics.effective_batch_size_p50 = float(np.percentile(arr, 50))
            self._metrics.effective_batch_size_p95 = float(np.percentile(arr, 95))
            self._metrics.effective_batch_size_p99 = float(np.percentile(arr, 99))
        else:
            self._metrics.effective_batch_size_mean = 0.0
            self._metrics.effective_batch_size_p50 = 0.0
            self._metrics.effective_batch_size_p95 = 0.0
            self._metrics.effective_batch_size_p99 = 0.0

        start_ns = self._run_start_wall_ns or 0
        end_ns = self._run_end_wall_ns or 0
        # Clamp wall-clock to 1 nanosecond to avoid divide-by-zero on
        # zero-event runs; goodput on such runs is meaningless but
        # finite.
        wall_s = max((end_ns - start_ns) / 1e9, 1e-9)
        self._metrics.goodput_requests_per_second = self._total_completed / wall_s

        if self._total_submitted > 0:
            self._metrics.completion_ratio = self._total_completed / self._total_submitted
        else:
            self._metrics.completion_ratio = 0.0

        if self._first_oom_wall_ns is not None and self._run_start_wall_ns is not None:
            self._metrics.time_to_first_oom_seconds = (
                self._first_oom_wall_ns - self._run_start_wall_ns
            ) / 1e9
        else:
            self._metrics.time_to_first_oom_seconds = None

    def sample(self, num_active_requests: int) -> None:
        """Capture an occupancy snapshot for the run timeline."""

        self._metrics.active_requests_samples.append(num_active_requests)
        if self._device == "cuda":
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
        else:
            astats = self._allocator.stats()
            allocated = astats.allocated_blocks
            reserved = astats.total_blocks
        ratio = 1.0 - (allocated / reserved) if reserved > 0 else 0.0
        self._metrics.fragmentation_samples.append(ratio)

    @property
    def metrics(self) -> AllocatorMetrics:
        return self._metrics


def _record_memory_history_enable(max_entries: int) -> None:
    """Enable PyTorch's allocator history recorder.

    Wraps the underscore-prefixed but documented API
    ``torch.cuda.memory._record_memory_history``. See the PyTorch blog
    "Understanding GPU Memory 1".
    """

    fn = getattr(torch.cuda.memory, "_record_memory_history", None)
    if fn is None:
        raise RuntimeError("torch.cuda.memory._record_memory_history is unavailable")
    fn(enabled="all", max_entries=max_entries)


def _record_memory_history_dump(path: str) -> None:
    """Dump the recorded allocator history pickle to ``path``."""

    fn = getattr(torch.cuda.memory, "_dump_snapshot", None)
    if fn is None:
        raise RuntimeError("torch.cuda.memory._dump_snapshot is unavailable")
    fn(path)
    disable_fn = getattr(torch.cuda.memory, "_record_memory_history", None)
    if disable_fn is not None:
        disable_fn(enabled=None)
