#!/usr/bin/env python
"""Emit deterministic JSONL cache-planning probe records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cachepawl.bench.environment import capture_environment
from cachepawl.bench.synthetic_workloads import (
    DEFAULT_TIMESTAMP,
    PLANNER_BACKENDS,
    SYNTHETIC_WORKLOADS,
    ProbeBackend,
    SyntheticWorkloadName,
    build_probe_result,
    generate_synthetic_workload,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=SYNTHETIC_WORKLOADS, action="append")
    parser.add_argument("--backend", choices=PLANNER_BACKENDS, action="append")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-total-bytes", type=int)
    parser.add_argument("--gpu-name", default="NVIDIA GeForce RTX 3060")
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP)
    parser.add_argument(
        "--measure-runtime",
        action="store_true",
        help="record wall-clock planner_runtime_us; disabled by default for deterministic JSONL",
    )
    args = parser.parse_args(argv)

    workloads: list[SyntheticWorkloadName] = args.workload or list(SYNTHETIC_WORKLOADS)
    backends: list[ProbeBackend] = args.backend or list(PLANNER_BACKENDS)
    environment = capture_environment(
        total_memory_bytes_override=args.gpu_total_bytes,
        name_override=args.gpu_name,
    )
    lines: list[str] = []
    for workload_name in workloads:
        workload = generate_synthetic_workload(
            workload_name,
            seed=args.seed,
            num_requests=args.num_requests,
        )
        for backend in backends:
            result = build_probe_result(
                backend=backend,
                workload=workload,
                environment=environment,
                timestamp=args.timestamp,
                measure_runtime=args.measure_runtime,
            )
            lines.append(result.to_json_line())

    text = "\n".join(lines) + ("\n" if lines else "")
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
