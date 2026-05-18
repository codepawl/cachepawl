"""Tier 1 PR B: throughput-proxy metric formula and finalization tests.

These tests cover the seven new fields on :class:`AllocatorMetrics`
introduced with schema 1.2.0:

- effective_batch_size_{mean,p50,p95,p99}: derived from
  ``active_requests_samples`` filtered to positive entries.
- goodput_requests_per_second: completed / wall_clock_seconds.
- completion_ratio: completed / submitted under the strict
  semantic (no OOM during request lifetime AND clean free).
- time_to_first_oom_seconds: wall-clock lead time to first OOM;
  None when the run had no OOMs.

The formulas are exercised through :class:`MetricsCollector` so the
tests cover the call sequence the runner actually performs.
"""

from __future__ import annotations

import time

import pytest
import torch

from cachepawl.benchmarks import (
    MetricsCollector,
)
from tests.unit.benchmarks.conftest import FakeAllocator


def test_effective_batch_size_filters_zero_samples() -> None:
    """Samples with active==0 are dropped; statistics use only positive ticks.

    The mean over [3, 5, 7] is 5.0; the p50 (numpy linear interpolation)
    is 5.0; p95 over those three values evaluates to 6.8; p99 to 6.96.
    The two leading zeros and trailing zero are filtered out, modeling
    the runner's idle-tick and post-teardown sample.
    """

    with MetricsCollector(device="cpu", allocator=FakeAllocator()) as collector:
        # Populate active_requests_samples by hand. We do NOT exercise the
        # runner here; that's covered by the smoke test below.
        for x in [0, 0, 3, 5, 7, 0]:
            collector.metrics.active_requests_samples.append(x)
    collector.finalize_throughput_metrics()

    metrics = collector.metrics
    assert metrics.effective_batch_size_mean == pytest.approx(5.0)
    assert metrics.effective_batch_size_p50 == pytest.approx(5.0)
    assert metrics.effective_batch_size_p95 == pytest.approx(6.8)
    assert metrics.effective_batch_size_p99 == pytest.approx(6.96)


def test_effective_batch_size_all_zero_samples() -> None:
    """All-zero filtered series leaves the four fields at 0.0 default."""

    with MetricsCollector(device="cpu", allocator=FakeAllocator()) as collector:
        for _ in range(3):
            collector.metrics.active_requests_samples.append(0)
    collector.finalize_throughput_metrics()

    metrics = collector.metrics
    assert metrics.effective_batch_size_mean == 0.0
    assert metrics.effective_batch_size_p50 == 0.0
    assert metrics.effective_batch_size_p95 == 0.0
    assert metrics.effective_batch_size_p99 == 0.0


def test_completion_ratio_clean_run() -> None:
    """submitted=10, completed=10 -> 1.0 (no OOM rejections)."""

    with MetricsCollector(device="cpu", allocator=FakeAllocator()) as collector:
        for _ in range(10):
            collector.record_submitted()
            collector.record_completed()
    collector.finalize_throughput_metrics()

    assert collector.metrics.completion_ratio == 1.0


def test_completion_ratio_partial_oom() -> None:
    """submitted=10, completed=7 -> 0.7. Models 3 requests that OOM'd."""

    with MetricsCollector(device="cpu", allocator=FakeAllocator()) as collector:
        for _ in range(10):
            collector.record_submitted()
        for _ in range(7):
            collector.record_completed()
    collector.finalize_throughput_metrics()

    assert collector.metrics.completion_ratio == pytest.approx(0.7)


def test_completion_ratio_zero_submitted_returns_zero() -> None:
    """An empty run reports 0.0 instead of dividing by zero."""

    with MetricsCollector(device="cpu", allocator=FakeAllocator()) as collector:
        pass
    collector.finalize_throughput_metrics()

    assert collector.metrics.completion_ratio == 0.0


def test_goodput_basic_division() -> None:
    """completed=100, wall=~10ms -> ~10_000 req/sec, modulo timer noise."""

    with MetricsCollector(device="cpu", allocator=FakeAllocator()) as collector:
        for _ in range(100):
            collector.record_completed()
        # Sleep ~10ms inside the with block so end - start spans real time.
        time.sleep(0.01)
    collector.finalize_throughput_metrics()

    metrics = collector.metrics
    # Wall clock between __enter__ and __exit__ is at least 10 ms, so
    # goodput is bounded above by 100 / 0.01 = 10_000 req/s. The lower
    # bound is loose to absorb timer jitter on slow CI hosts.
    assert 10.0 < metrics.goodput_requests_per_second <= 10_500.0


def test_goodput_finite_on_unset_wall_clock() -> None:
    """finalize without an open MetricsCollector window does not divide by zero."""

    collector = MetricsCollector(device="cpu", allocator=FakeAllocator())
    collector.record_completed()
    # No __enter__ / __exit__ ever called; both wall stamps stay None.
    collector.finalize_throughput_metrics()
    # The 1e-9 clamp keeps goodput finite (it will be huge, but never inf).
    assert collector.metrics.goodput_requests_per_second != float("inf")
    assert collector.metrics.goodput_requests_per_second > 0.0


def test_time_to_first_oom_none_when_no_ooms() -> None:
    """A run with zero record_oom calls reports None."""

    with MetricsCollector(device="cpu", allocator=FakeAllocator()) as collector:
        collector.record_completed()
    collector.finalize_throughput_metrics()

    assert collector.metrics.time_to_first_oom_seconds is None


def test_time_to_first_oom_captures_first_call_only() -> None:
    """Two record_oom calls record only the first wall-clock stamp.

    Subsequent OOMs increment oom_count but must not overwrite the
    captured lead time; otherwise the metric would lose its
    pressure-onset semantics.
    """

    with MetricsCollector(device="cpu", allocator=FakeAllocator()) as collector:
        collector.record_oom()
        first_stamp = collector._first_oom_wall_ns
        time.sleep(0.005)
        collector.record_oom()
        assert collector._first_oom_wall_ns == first_stamp
    collector.finalize_throughput_metrics()

    ttfo = collector.metrics.time_to_first_oom_seconds
    assert ttfo is not None
    assert ttfo >= 0.0
    assert collector.metrics.oom_count == 2


def test_finalize_is_idempotent() -> None:
    """Calling finalize_throughput_metrics twice yields the same values."""

    with MetricsCollector(device="cpu", allocator=FakeAllocator()) as collector:
        for x in [0, 4, 4, 0]:
            collector.metrics.active_requests_samples.append(x)
        for _ in range(2):
            collector.record_submitted()
        collector.record_completed()
    collector.finalize_throughput_metrics()
    first = (
        collector.metrics.effective_batch_size_p50,
        collector.metrics.completion_ratio,
        collector.metrics.goodput_requests_per_second,
    )
    collector.finalize_throughput_metrics()
    second = (
        collector.metrics.effective_batch_size_p50,
        collector.metrics.completion_ratio,
        collector.metrics.goodput_requests_per_second,
    )
    assert first == second


class _OomOnceAllocator(FakeAllocator):
    """FakeAllocator that raises OOM on the Nth allocate call.

    Used to model strict completion semantics: a request that partially
    OOMs during arrival must not count as completed even if its (empty)
    free at departure succeeds.
    """

    def __init__(self, *, oom_on_call: int, total_blocks: int = 100_000) -> None:
        super().__init__(total_blocks=total_blocks)
        self._oom_on_call = oom_on_call

    def allocate(self, num_blocks: int, *, dtype_bytes: int) -> list[int]:
        if self.allocate_calls + 1 == self._oom_on_call:
            self.allocate_calls += 1
            raise torch.cuda.OutOfMemoryError("synthetic OOM for strict-completion test")
        return super().allocate(num_blocks, dtype_bytes=dtype_bytes)


def test_runner_strict_completion_excludes_partial_oom_request() -> None:
    """End-to-end: a request whose allocate OOMs is submitted but NOT completed.

    The runner threads request_id and request_had_oom through
    _timed_allocate; the strict completion rule in _process_departure
    must drop the OOM'd request from the completed count even though
    its departure event still fires.
    """

    from dataclasses import replace
    from pathlib import Path

    from cachepawl.benchmarks import PRESETS, run_benchmark

    spec = replace(PRESETS["uniform_short"], num_requests=4, seed=1)
    # OOM on call #2: the second allocate (could be either the ssm leg
    # of request 0 or the kv leg of request 1, depending on the
    # arrival sequence). Either way, the OOM'd request's lifetime
    # carries the flag and it must NOT count as completed.
    allocator = _OomOnceAllocator(oom_on_call=2)
    output_dir = Path("/tmp/cp-strict-completion-test")

    run = run_benchmark(
        allocator=allocator,
        spec=spec,
        allocator_name="strict_completion_test",
        output_dir=output_dir,
        device="cpu",
    )
    assert run.metrics.oom_count >= 1
    # 4 requests submitted; at most 3 can have completed cleanly
    # because one had an OOM during its lifetime.
    assert run.metrics.completion_ratio <= 0.75


def test_runner_smoke_populates_all_seven_throughput_fields() -> None:
    """End-to-end: run_benchmark on a tiny workload sets every new field.

    Asserts types and ranges, not exact values: this is a smoke test
    against silent regression where finalize_throughput_metrics is no
    longer called or a field is left at its default.
    """

    from dataclasses import replace
    from pathlib import Path

    from cachepawl.benchmarks import PRESETS, run_benchmark

    spec = replace(PRESETS["uniform_short"], num_requests=8, seed=42)
    run = run_benchmark(
        allocator=FakeAllocator(),
        spec=spec,
        allocator_name="throughput_smoke",
        output_dir=Path("/tmp/cp-throughput-fields-smoke"),
        device="cpu",
    )

    m = run.metrics
    assert isinstance(m.effective_batch_size_mean, float)
    assert isinstance(m.effective_batch_size_p50, float)
    assert isinstance(m.effective_batch_size_p95, float)
    assert isinstance(m.effective_batch_size_p99, float)
    assert isinstance(m.goodput_requests_per_second, float)
    assert isinstance(m.completion_ratio, float)
    assert m.time_to_first_oom_seconds is None or isinstance(m.time_to_first_oom_seconds, float)
    # FakeAllocator never OOMs and every request fully completes.
    assert m.completion_ratio == 1.0
    assert m.time_to_first_oom_seconds is None
    # p50 in [1, 8] because there are at most 8 active requests at once.
    assert 0.0 < m.effective_batch_size_p50 <= 8.0
