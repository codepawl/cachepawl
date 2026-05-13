"""Benchmark harness public surface and allocator registry.

The registry is intentionally empty in this PR. Future allocator PRs append
factory callables under unique names; the CLI in ``run.py`` rejects unknown
names with the list of registered ones.
"""

from collections.abc import Callable

from cachepawl.allocator.base import Allocator
from cachepawl.benchmarks.harness.metrics import (
    AllocatorMetrics,
    LatencyPercentiles,
    MetricsCollector,
    compute_percentiles,
)
from cachepawl.benchmarks.harness.runner import run_benchmark
from cachepawl.benchmarks.harness.schema import (
    SCHEMA_VERSION,
    BenchmarkRun,
    Environment,
    Hardware,
)
from cachepawl.benchmarks.harness.workloads import (
    JAMBA_MINI_ATTN,
    JAMBA_MINI_SSM,
    PRESETS,
    AttentionLayerProfile,
    Request,
    SSMLayerProfile,
    WorkloadSpec,
    generate_request_stream,
    per_sequence_ssm_bytes,
    per_token_kv_bytes,
)

AllocatorFactory = Callable[[], Allocator]
REGISTRY: dict[str, AllocatorFactory] = {}

__all__ = [
    "JAMBA_MINI_ATTN",
    "JAMBA_MINI_SSM",
    "PRESETS",
    "REGISTRY",
    "SCHEMA_VERSION",
    "AllocatorFactory",
    "AllocatorMetrics",
    "AttentionLayerProfile",
    "BenchmarkRun",
    "Environment",
    "Hardware",
    "LatencyPercentiles",
    "MetricsCollector",
    "Request",
    "SSMLayerProfile",
    "WorkloadSpec",
    "compute_percentiles",
    "generate_request_stream",
    "per_sequence_ssm_bytes",
    "per_token_kv_bytes",
    "run_benchmark",
]
