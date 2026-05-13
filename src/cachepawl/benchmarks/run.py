"""Command-line entrypoint for the benchmark harness.

Usage:
    python -m cachepawl.benchmarks.run \\
        --workload {uniform_short,mixed_long,agentic_burst} \\
        --allocator <name> \\
        [--device {cpu,cuda}] \\
        [--output benchmarks/results/] \\
        [--seed N] \\
        [--record-memory-snapshot] \\
        [--notes "free-form string"]

Exit codes:
    0  success
    2  unknown workload or allocator name
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

from cachepawl.benchmarks import PRESETS, REGISTRY, run_benchmark
from cachepawl.utils.device import get_device


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    workload_name: str = args.workload
    allocator_name: str = args.allocator
    device: str = args.device or get_device()
    output_dir = Path(args.output)
    seed_override: int | None = args.seed
    notes: str = args.notes
    record_memory_snapshot: bool = args.record_memory_snapshot

    if workload_name not in PRESETS:
        _print_unknown("workload", workload_name, sorted(PRESETS))
        return 2

    if allocator_name not in REGISTRY:
        if not REGISTRY:
            print(
                f"error: Unknown allocator {allocator_name!r}: "
                "no allocators are registered yet (this PR ships the harness only).",
                file=sys.stderr,
            )
        else:
            _print_unknown("allocator", allocator_name, sorted(REGISTRY))
        return 2

    spec = PRESETS[workload_name]
    if seed_override is not None:
        spec = dataclasses.replace(spec, seed=seed_override)

    factory = REGISTRY[allocator_name]
    allocator = factory()
    run = run_benchmark(
        allocator=allocator,
        spec=spec,
        allocator_name=allocator_name,
        output_dir=output_dir,
        device=device,
        notes=notes,
        record_memory_snapshot=record_memory_snapshot,
    )
    print(
        f"wrote BenchmarkRun: allocator={run.allocator_name} "
        f"workload={run.spec.name} schema_version={run.schema_version}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cachepawl.benchmarks.run",
        description="Run a model-free micro-benchmark against a registered Allocator.",
    )
    parser.add_argument("--workload", required=True, help="preset name")
    parser.add_argument("--allocator", required=True, help="registered allocator name")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="device to run on; defaults to cachepawl.utils.device.get_device()",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results",
        help="root output directory for JSON results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the preset seed for one-off reruns",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="free-form string written into BenchmarkRun.notes",
    )
    parser.add_argument(
        "--record-memory-snapshot",
        action="store_true",
        help="enable torch.cuda.memory snapshot recording (cuda only)",
    )
    return parser


def _print_unknown(label: str, given: str, registered: list[str]) -> None:
    print(
        f"error: Unknown {label} {given!r}. Registered {label}s: {registered}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
