#!/usr/bin/env python
"""Compare vLLM-style padded planning against Cachepawl AVMP planning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cachepawl.bench.environment import capture_environment
from cachepawl.bench.planner_baselines import COMPARISON_BACKENDS, PlannerBackend
from cachepawl.bench.planner_comparison import (
    compare_planners,
    render_csv_summary,
    render_jsonl,
    render_markdown_summary,
)
from cachepawl.bench.synthetic_workloads import (
    DEFAULT_TIMESTAMP,
    SYNTHETIC_WORKLOADS,
    SyntheticWorkloadName,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=SYNTHETIC_WORKLOADS, action="append")
    parser.add_argument("--backend", choices=COMPARISON_BACKENDS, action="append")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--gpu-total-bytes", type=int)
    parser.add_argument("--gpu-name", default="NVIDIA GeForce RTX 3060")
    parser.add_argument("--jsonl-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--summary-format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP)
    parser.add_argument(
        "--measure-runtime",
        action="store_true",
        help="record wall-clock planner_runtime_us; disabled by default for deterministic output",
    )
    args = parser.parse_args(argv)

    workloads: tuple[SyntheticWorkloadName, ...] = tuple(args.workload or SYNTHETIC_WORKLOADS)
    backends: tuple[PlannerBackend, ...] = tuple(args.backend or COMPARISON_BACKENDS)
    environment = capture_environment(
        total_memory_bytes_override=args.gpu_total_bytes,
        name_override=args.gpu_name,
    )
    records = compare_planners(
        workloads=workloads,
        backends=backends,
        seed=args.seed,
        num_requests=args.num_requests,
        environment=environment,
        timestamp=args.timestamp,
        measure_runtime=args.measure_runtime,
    )

    if args.jsonl_output is not None:
        args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl_output.write_text(render_jsonl(records))

    summary = (
        render_markdown_summary(records)
        if args.summary_format == "markdown"
        else render_csv_summary(records)
    )
    if args.summary_output is None:
        sys.stdout.write(summary)
    else:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
