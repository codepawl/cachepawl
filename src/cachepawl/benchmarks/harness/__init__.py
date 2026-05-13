"""Internal building blocks of the benchmark harness.

End users should import from ``cachepawl.benchmarks`` instead of this
package; everything here is re-exported there.
"""

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

__all__ = [
    "JAMBA_MINI_ATTN",
    "JAMBA_MINI_SSM",
    "PRESETS",
    "SCHEMA_VERSION",
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
