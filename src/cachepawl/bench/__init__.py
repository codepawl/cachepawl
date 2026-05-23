"""Lightweight cache-planning benchmark spine.

This package is separate from ``cachepawl.benchmarks``. The latter runs
allocator simulations; this package emits planner-level records for
baseline memory-efficiency tables before runtime vLLM integration exists.
"""

from cachepawl.bench.environment import (
    RuntimeEnvironment,
    capture_environment,
    capture_gpu_metadata,
)
from cachepawl.bench.planner_baselines import (
    COMPARISON_BACKENDS,
    PlannerBackend,
    PlannerEstimate,
    estimate_avmp_static,
    estimate_planner,
    estimate_vllm_style_padded,
)
from cachepawl.bench.planner_comparison import (
    compare_planners,
    render_csv_summary,
    render_jsonl,
    render_markdown_summary,
)
from cachepawl.bench.result_schema import (
    BENCH_RESULT_SCHEMA_VERSION,
    CacheProbeResult,
    GpuMetadata,
    Metadata,
    MetadataValue,
)
from cachepawl.bench.synthetic_workloads import (
    PLANNER_BACKENDS,
    SYNTHETIC_WORKLOADS,
    ProbeBackend,
    SyntheticRequest,
    SyntheticWorkload,
    build_probe_result,
    generate_synthetic_workload,
)

__all__ = [
    "BENCH_RESULT_SCHEMA_VERSION",
    "COMPARISON_BACKENDS",
    "PLANNER_BACKENDS",
    "SYNTHETIC_WORKLOADS",
    "CacheProbeResult",
    "GpuMetadata",
    "Metadata",
    "MetadataValue",
    "PlannerBackend",
    "PlannerEstimate",
    "ProbeBackend",
    "RuntimeEnvironment",
    "SyntheticRequest",
    "SyntheticWorkload",
    "build_probe_result",
    "capture_environment",
    "capture_gpu_metadata",
    "compare_planners",
    "estimate_avmp_static",
    "estimate_planner",
    "estimate_vllm_style_padded",
    "generate_synthetic_workload",
    "render_csv_summary",
    "render_jsonl",
    "render_markdown_summary",
]
