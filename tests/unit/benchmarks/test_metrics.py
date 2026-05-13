"""Tests for the metrics collector and percentile helpers."""

from __future__ import annotations

import pytest

from cachepawl.benchmarks import (
    AllocatorMetrics,
    LatencyPercentiles,
    MetricsCollector,
    compute_percentiles,
)
from tests.unit.benchmarks.conftest import FakeAllocator


def test_compute_percentiles_on_known_distribution() -> None:
    samples = list(range(1, 101))
    p = compute_percentiles(samples)
    assert p.p50_ns == 50
    assert p.p95_ns == 95
    assert p.p99_ns == 99
    assert p.max_ns == 100


def test_compute_percentiles_on_empty_list_returns_zeros() -> None:
    p = compute_percentiles([])
    assert p == LatencyPercentiles(p50_ns=0, p95_ns=0, p99_ns=0, max_ns=0)


def test_metrics_records_allocate_and_free_latencies() -> None:
    metrics = AllocatorMetrics()
    metrics.allocate_latency_ns.extend([100, 200, 300])
    metrics.free_latency_ns.extend([50, 60])
    alloc = metrics.allocate_latency_percentiles()
    free = metrics.free_latency_percentiles()
    assert alloc.max_ns == 300
    assert free.max_ns == 60


def test_collector_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        MetricsCollector(device="rocm", allocator=FakeAllocator())


def test_collector_requires_snapshot_path_when_recording() -> None:
    with pytest.raises(ValueError, match="snapshot_path is required"):
        MetricsCollector(
            device="cuda",
            allocator=FakeAllocator(),
            record_memory_snapshot=True,
            snapshot_path=None,
        )


def test_collector_cpu_path_records_via_allocator_stats() -> None:
    allocator = FakeAllocator(total_blocks=1000)
    with MetricsCollector(device="cpu", allocator=allocator) as collector:
        allocator.allocate(10, dtype_bytes=2)
        collector.record_allocate(123)
        collector.sample(num_active_requests=1)
        allocator.allocate(40, dtype_bytes=2)
        collector.sample(num_active_requests=2)
    metrics = collector.metrics
    assert metrics.allocate_latency_ns == [123]
    assert metrics.active_requests_samples == [1, 2]
    assert len(metrics.fragmentation_samples) == 2
    assert metrics.fragmentation_samples[0] == pytest.approx(1.0 - 10 / 1000)
    assert metrics.fragmentation_samples[1] == pytest.approx(1.0 - 50 / 1000)
    assert metrics.peak_allocated_bytes == 50
    assert metrics.peak_reserved_bytes == 1000


def test_collector_records_oom_and_preemption() -> None:
    with MetricsCollector(device="cpu", allocator=FakeAllocator()) as collector:
        collector.record_oom()
        collector.record_oom()
        collector.record_preemption()
    assert collector.metrics.oom_count == 2
    assert collector.metrics.preemption_count == 1


def test_collector_fragmentation_safe_on_empty_pool() -> None:
    allocator = FakeAllocator(total_blocks=0)
    with MetricsCollector(device="cpu", allocator=allocator) as collector:
        collector.sample(num_active_requests=0)
    assert collector.metrics.fragmentation_samples == [0.0]
