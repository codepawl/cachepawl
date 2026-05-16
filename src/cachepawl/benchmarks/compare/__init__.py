"""Comparison sweep runner.

Public surface for sweeping registered baseline allocators across the
workload x model-spec x pool-size grid, aggregating replicates, and
rendering reports plus matplotlib plots.

Invoke the CLI via ``python -m cachepawl.benchmarks.compare``.
"""

from cachepawl.benchmarks.compare.aggregate import (
    AggregatedMetrics,
    AggregatedRow,
    aggregate_runs,
    compute_relative_improvement,
)
from cachepawl.benchmarks.compare.plots import (
    plot_fragmentation_vs_workload,
    plot_oom_count_vs_workload,
    plot_padding_waste_vs_state_size,
)
from cachepawl.benchmarks.compare.report import (
    render_deterministic_summary,
    render_json_summary,
    render_markdown_report,
)
from cachepawl.benchmarks.compare.sweep import (
    DEFAULT_MODEL_SPEC_NAMES,
    DEFAULT_SEED_REPLICATES,
    DEFAULT_TOTAL_BYTES_OPTIONS,
    DEFAULT_VARIANTS,
    DEFAULT_WORKLOAD_NAMES,
    QUICK_MODEL_SPEC_NAMES,
    QUICK_SEED_REPLICATES,
    QUICK_TOTAL_BYTES_OPTIONS,
    QUICK_WORKLOAD_NAMES,
    SMOKE_NUM_REQUESTS,
    AllocatorVariant,
    CellFailure,
    SweepConfig,
    SweepMetadata,
    SweepResult,
    get_model_spec,
    known_model_spec_names,
    main,
    make_default_config,
    make_quick_config,
    make_smoke_config,
    run_sweep,
    total_bytes_human,
)

__all__ = [
    "DEFAULT_MODEL_SPEC_NAMES",
    "DEFAULT_SEED_REPLICATES",
    "DEFAULT_TOTAL_BYTES_OPTIONS",
    "DEFAULT_VARIANTS",
    "DEFAULT_WORKLOAD_NAMES",
    "QUICK_MODEL_SPEC_NAMES",
    "QUICK_SEED_REPLICATES",
    "QUICK_TOTAL_BYTES_OPTIONS",
    "QUICK_WORKLOAD_NAMES",
    "SMOKE_NUM_REQUESTS",
    "AggregatedMetrics",
    "AggregatedRow",
    "AllocatorVariant",
    "CellFailure",
    "SweepConfig",
    "SweepMetadata",
    "SweepResult",
    "aggregate_runs",
    "compute_relative_improvement",
    "get_model_spec",
    "known_model_spec_names",
    "main",
    "make_default_config",
    "make_quick_config",
    "make_smoke_config",
    "plot_fragmentation_vs_workload",
    "plot_oom_count_vs_workload",
    "plot_padding_waste_vs_state_size",
    "render_deterministic_summary",
    "render_json_summary",
    "render_markdown_report",
    "run_sweep",
    "total_bytes_human",
]
