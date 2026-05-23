#!/usr/bin/env python
"""Capture a measurement-only vanilla vLLM runtime baseline record."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from hashlib import sha256
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from importlib.util import find_spec
from pathlib import Path

from cachepawl.bench.environment import RuntimeEnvironment, capture_environment
from cachepawl.bench.result_schema import BENCH_RESULT_SCHEMA_VERSION, CacheProbeResult
from cachepawl.bench.synthetic_workloads import DEFAULT_TIMESTAMP

DEFAULT_OUTPUT_DIR = Path("research/avmp/v2/results/vllm-baseline")
DEFAULT_MODEL = "Zyphra/Zamba2-2.7B-instruct"
DEFAULT_FALLBACK_MODEL = "tiiuae/Falcon-H1-1.5B-Instruct"
PINNED_VLLM_VERSION = "0.21.0"
PINNED_VENV_PATH = "/tmp/vllm-cachepawl-venv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL)
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-total-bytes", type=int, default=12 * 1024**3)
    parser.add_argument("--gpu-name", default="NVIDIA GeForce RTX 3060")
    args = parser.parse_args(argv)

    capture_baseline(
        output_dir=args.output_dir,
        model=args.model,
        fallback_model=args.fallback_model,
        timestamp=args.timestamp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        gpu_total_bytes=args.gpu_total_bytes,
        gpu_name=args.gpu_name,
    )
    return 0


def capture_baseline(
    *,
    output_dir: Path,
    model: str,
    fallback_model: str,
    timestamp: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    gpu_total_bytes: int,
    gpu_name: str,
) -> CacheProbeResult:
    """Write the current vanilla vLLM runtime baseline status."""

    output_dir.mkdir(parents=True, exist_ok=True)
    environment = capture_environment(
        total_memory_bytes_override=gpu_total_bytes,
        name_override=gpu_name,
    )
    vllm_installed = find_spec("vllm") is not None
    vllm_version = _optional_package_version("vllm")
    nvidia_smi = _capture_nvidia_smi()
    status, reason = _runtime_status(
        vllm_installed=vllm_installed,
        cuda_available=environment.gpu.cuda_available,
        nvidia_smi_status=nvidia_smi["status"],
    )
    result = _not_runnable_result(
        timestamp=timestamp,
        model=model,
        fallback_model=fallback_model,
        environment=environment,
        vllm_installed=vllm_installed,
        vllm_version=vllm_version,
        status=status,
        reason=reason,
        nvidia_smi=nvidia_smi,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
    )
    (output_dir / "baseline.jsonl").write_text(result.to_json_line() + "\n")
    generation_command = _generation_command(
        output_dir=output_dir,
        model=model,
        fallback_model=fallback_model,
        timestamp=timestamp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        gpu_total_bytes=gpu_total_bytes,
        gpu_name=gpu_name,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            _manifest_payload(
                timestamp=timestamp,
                model=model,
                fallback_model=fallback_model,
                status=status,
                reason=reason,
                vllm_installed=vllm_installed,
                vllm_version=vllm_version,
                environment=environment,
                generation_command=generation_command,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_seqs=max_num_seqs,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "README.md").write_text(
        _readme(
            generation_command=generation_command,
            status=status,
            reason=reason,
            model=model,
            fallback_model=fallback_model,
        )
    )
    return result


def _not_runnable_result(
    *,
    timestamp: str,
    model: str,
    fallback_model: str,
    environment: RuntimeEnvironment,
    vllm_installed: bool,
    vllm_version: str | None,
    status: str,
    reason: str,
    nvidia_smi: dict[str, str | int | None],
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
) -> CacheProbeResult:
    metadata = {
        **environment.metadata,
        "status": status,
        "reason": reason,
        "vllm_installed": vllm_installed,
        "vllm_version": vllm_version,
        "pinned_vllm_version": PINNED_VLLM_VERSION,
        "pinned_venv_path": PINNED_VENV_PATH,
        "fallback_model": fallback_model,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "cuda_available": environment.gpu.cuda_available,
        "nvidia_smi_status": nvidia_smi["status"],
        "nvidia_smi_returncode": nvidia_smi["returncode"],
        "nvidia_smi_stdout": nvidia_smi["stdout"],
        "nvidia_smi_stderr": nvidia_smi["stderr"],
        "measurement_kind": "vanilla-vllm-runtime-baseline",
        "allocator_replacement": False,
        "monkeypatching": False,
    }
    return CacheProbeResult(
        run_id=_run_id(timestamp=timestamp, model=model, status=status, reason=reason),
        timestamp=timestamp,
        backend="vllm-runtime-baseline",
        workload="runtime-baseline",
        model=model,
        gpu=environment.gpu,
        estimated_bytes=0,
        reserved_bytes=0,
        useful_bytes=0,
        overestimation_ratio=0.0,
        wasted_fraction=0.0,
        virtual_oom=False,
        planner_runtime_us=0.0,
        metadata=metadata,
    )


def _runtime_status(
    *,
    vllm_installed: bool,
    cuda_available: bool,
    nvidia_smi_status: str | int | None,
) -> tuple[str, str]:
    if not vllm_installed:
        return "not_runnable", "vllm is not installed in the active Python environment"
    if not cuda_available:
        return "not_runnable", "torch reports CUDA unavailable"
    if nvidia_smi_status != "ok":
        return "not_runnable", "nvidia-smi did not complete successfully"
    return "ready", "vanilla vLLM baseline command can be attempted manually"


def _capture_nvidia_smi() -> dict[str, str | int | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.free,memory.total",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return {
            "status": "not_found",
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[:500],
        "stderr": completed.stderr.strip()[:500],
    }


def _manifest_payload(
    *,
    timestamp: str,
    model: str,
    fallback_model: str,
    status: str,
    reason: str,
    vllm_installed: bool,
    vllm_version: str | None,
    environment: RuntimeEnvironment,
    generation_command: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
) -> dict[str, object]:
    return {
        "artifact_name": "vllm-runtime-baseline",
        "generated_at": timestamp,
        "status": status,
        "reason": reason,
        "schema_version": BENCH_RESULT_SCHEMA_VERSION,
        "model": model,
        "fallback_model": fallback_model,
        "pinned_vllm_version": PINNED_VLLM_VERSION,
        "pinned_venv_path": PINNED_VENV_PATH,
        "vllm_installed": vllm_installed,
        "vllm_version": vllm_version,
        "cuda_available": environment.gpu.cuda_available,
        "gpu": environment.gpu.to_dict(),
        "runtime": dict(environment.metadata),
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "generation_command": generation_command,
    }


def _readme(
    *,
    generation_command: str,
    status: str,
    reason: str,
    model: str,
    fallback_model: str,
) -> str:
    return f"""# vLLM Runtime Baseline Capture

This directory records the measurement-only vanilla vLLM baseline status for
Sprint 1 / T001.

## Contents

- `baseline.jsonl`
- `manifest.json`
- `README.md`

## Reproduction

```bash
{generation_command}
```

## Status

- status: `{status}`
- reason: `{reason}`
- primary model: `{model}`
- fallback model: `{fallback_model}`
- pinned vLLM: `{PINNED_VLLM_VERSION}`
- isolated venv: `{PINNED_VENV_PATH}`

This step does not install vLLM, serve a model, monkeypatch vLLM, replace
allocators, add Triton kernels, add copy kernels, add LSDR, or run real model
quality evaluation.
"""


def _generation_command(
    *,
    output_dir: Path,
    model: str,
    fallback_model: str,
    timestamp: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    gpu_total_bytes: int,
    gpu_name: str,
) -> str:
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python "
        "benchmarks/scripts/capture_vllm_baseline.py "
        f"--output-dir {output_dir.as_posix()} "
        f"--model {json.dumps(model)} "
        f"--fallback-model {json.dumps(fallback_model)} "
        f"--timestamp {json.dumps(timestamp)} "
        f"--max-model-len {max_model_len} "
        f"--gpu-memory-utilization {gpu_memory_utilization} "
        f"--max-num-seqs {max_num_seqs} "
        f"--gpu-total-bytes {gpu_total_bytes} "
        f"--gpu-name {json.dumps(gpu_name)}"
    )


def _optional_package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _run_id(*, timestamp: str, model: str, status: str, reason: str) -> str:
    material = f"{timestamp}|{model}|{status}|{reason}"
    return sha256(material.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
