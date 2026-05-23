#!/usr/bin/env python
"""Create a reproducible planner-comparison artifact pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cachepawl.bench.environment import RuntimeEnvironment, capture_environment
from cachepawl.bench.planner_comparison import (
    compare_planners,
    render_jsonl,
    render_markdown_summary,
)
from cachepawl.bench.result_schema import BENCH_RESULT_SCHEMA_VERSION, CacheProbeResult
from cachepawl.bench.synthetic_workloads import DEFAULT_TIMESTAMP, SYNTHETIC_WORKLOADS
from cachepawl.models.spec import JAMBA_1_5_MINI_REF

DEFAULT_OUTPUT_DIR = Path("benchmarks/results/rtx3060/planner-comparison")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--gpu-total-bytes", type=int, default=12 * 1024**3)
    parser.add_argument("--gpu-name", default="NVIDIA GeForce RTX 3060")
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP)
    parser.add_argument(
        "--measure-runtime",
        action="store_true",
        help="record wall-clock planner_runtime_us; disabled by default for deterministic output",
    )
    args = parser.parse_args(argv)

    create_artifact_pack(
        output_dir=args.output_dir,
        seed=args.seed,
        num_requests=args.num_requests,
        gpu_total_bytes=args.gpu_total_bytes,
        gpu_name=args.gpu_name,
        timestamp=args.timestamp,
        measure_runtime=args.measure_runtime,
    )
    return 0


def create_artifact_pack(
    *,
    output_dir: Path,
    seed: int,
    num_requests: int,
    gpu_total_bytes: int,
    gpu_name: str,
    timestamp: str,
    measure_runtime: bool,
) -> None:
    """Write per-workload JSONL, combined summary, environment, and README files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    environment = capture_environment(
        total_memory_bytes_override=gpu_total_bytes,
        name_override=gpu_name,
    )
    all_records: list[CacheProbeResult] = []
    for workload in SYNTHETIC_WORKLOADS:
        records = compare_planners(
            workloads=(workload,),
            seed=seed,
            num_requests=num_requests,
            environment=environment,
            timestamp=timestamp,
            measure_runtime=measure_runtime,
        )
        (output_dir / f"{workload}.jsonl").write_text(render_jsonl(records))
        all_records.extend(records)

    (output_dir / "summary.md").write_text(render_markdown_summary(tuple(all_records)))
    (output_dir / "environment.json").write_text(
        json.dumps(
            _environment_payload(
                gpu_name=gpu_name,
                gpu_total_bytes=gpu_total_bytes,
                seed=seed,
                num_requests=num_requests,
                timestamp=timestamp,
                measure_runtime=measure_runtime,
                environment=environment,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    generation_command = _generation_command(
        seed=seed,
        num_requests=num_requests,
        gpu_name=gpu_name,
        gpu_total_bytes=gpu_total_bytes,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            _manifest_payload(
                seed=seed,
                num_requests=num_requests,
                gpu_name=gpu_name,
                gpu_total_bytes=gpu_total_bytes,
                timestamp=timestamp,
                measure_runtime=measure_runtime,
                generation_command=generation_command,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "README.md").write_text(
        _readme(
            seed=seed,
            num_requests=num_requests,
            gpu_name=gpu_name,
            gpu_total_bytes=gpu_total_bytes,
            timestamp=timestamp,
            measure_runtime=measure_runtime,
            generation_command=generation_command,
        )
    )


def _environment_payload(
    *,
    gpu_name: str,
    gpu_total_bytes: int,
    seed: int,
    num_requests: int,
    timestamp: str,
    measure_runtime: bool,
    environment: RuntimeEnvironment,
) -> dict[str, object]:
    return {
        "artifact": "planner-comparison",
        "model": JAMBA_1_5_MINI_REF.name,
        "workloads": list(SYNTHETIC_WORKLOADS),
        "backends": ["vllm-style-padded", "cachepawl-avmp"],
        "seed": seed,
        "num_requests": num_requests,
        "timestamp": timestamp,
        "measure_runtime": measure_runtime,
        "target_gpu": {
            "name": gpu_name,
            "total_memory_bytes": gpu_total_bytes,
        },
        "captured_gpu": environment.gpu.to_dict(),
        "runtime": dict(environment.metadata),
        "notes": [
            "vllm-style-padded is a modeling baseline, not exact vLLM internals.",
            "cachepawl-avmp is planner-only evidence, not runtime serving evidence.",
            "vLLM is not installed or required for this artifact.",
        ],
    }


def _manifest_payload(
    *,
    seed: int,
    num_requests: int,
    gpu_name: str,
    gpu_total_bytes: int,
    timestamp: str,
    measure_runtime: bool,
    generation_command: str,
) -> dict[str, object]:
    return {
        "artifact_name": "rtx3060-planner-comparison",
        "generated_at": timestamp,
        "seed": seed,
        "num_requests": num_requests,
        "workloads": list(SYNTHETIC_WORKLOADS),
        "backends": ["vllm-style-padded", "cachepawl-avmp"],
        "target_gpu_name": gpu_name,
        "target_gpu_total_bytes": gpu_total_bytes,
        "runtime_measurement_enabled": measure_runtime,
        "schema_version": BENCH_RESULT_SCHEMA_VERSION,
        "generation_command": generation_command,
    }


def _generation_command(
    *,
    seed: int,
    num_requests: int,
    gpu_name: str,
    gpu_total_bytes: int,
) -> str:
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python "
        "benchmarks/scripts/create_planner_comparison_pack.py "
        "--output-dir benchmarks/results/rtx3060/planner-comparison "
        f"--seed {seed} "
        f"--num-requests {num_requests} "
        f"--gpu-name {json.dumps(gpu_name)} "
        f"--gpu-total-bytes {gpu_total_bytes}"
    )


def _readme(
    *,
    seed: int,
    num_requests: int,
    gpu_name: str,
    gpu_total_bytes: int,
    timestamp: str,
    measure_runtime: bool,
    generation_command: str,
) -> str:
    return f"""# RTX 3060 Planner Comparison

This artifact pack is a deterministic planner-only comparison for synthetic
hybrid KV/State cache workloads.

## Contents

- `short-heavy.jsonl`
- `long-heavy.jsonl`
- `mixed.jsonl`
- `summary.md`
- `environment.json`
- `manifest.json`

## Reproduction

```bash
{generation_command}
```

Configuration:

- model: `{JAMBA_1_5_MINI_REF.name}`
- target GPU: `{gpu_name}`
- target GPU bytes: `{gpu_total_bytes}`
- seed: `{seed}`
- requests per workload: `{num_requests}`
- timestamp: `{timestamp}`
- runtime measurement: `{str(measure_runtime).lower()}`

## Interpretation

- `vllm-style-padded` is a modeling baseline for uniform padded cache planning,
  not an exact measurement of vLLM internals.
- `cachepawl-avmp` is planner-only evidence, not runtime vLLM serving evidence.
- `overestimation_ratio` is `estimated_bytes / useful_bytes`.
- `wasted_fraction` is `(estimated_bytes - useful_bytes) / estimated_bytes`.
- `planner_runtime_us` is deterministic `0.000` unless runtime measurement is
  explicitly enabled.
- AVMP can reduce overestimation while `virtual_oom` may still be true when the
  useful demand itself exceeds the 12GB target profile.

No vLLM install, runtime serving, monkeypatching, allocator replacement, Triton
kernels, copy kernels, LSDR, or real model inference are used for this pack.
"""


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
