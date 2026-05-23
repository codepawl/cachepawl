#!/usr/bin/env python
"""Capture a measurement-only vanilla vLLM runtime baseline record."""

from __future__ import annotations

import argparse
import json
import os
import shlex
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
INFRASTRUCTURE_DECISION = "fix-local-wsl2-gpu-nvml-first"
MetadataValue = str | int | float | bool | None
SmokeResult = dict[str, str | int | None]


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
    parser.add_argument(
        "--runtime-smoke",
        action="store_true",
        help="attempt a bounded vanilla vLLM model-load smoke; no serving or generation",
    )
    parser.add_argument("--runtime-timeout-seconds", type=int, default=1200)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--smoke-command", help=argparse.SUPPRESS)
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
        runtime_smoke=args.runtime_smoke,
        runtime_timeout_seconds=args.runtime_timeout_seconds,
        trust_remote_code=args.trust_remote_code,
        smoke_command=args.smoke_command,
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
    runtime_smoke: bool = False,
    runtime_timeout_seconds: int = 1200,
    trust_remote_code: bool = False,
    smoke_command: str | None = None,
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
    smoke_result = (
        _run_runtime_smoke(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            timeout_seconds=runtime_timeout_seconds,
            trust_remote_code=trust_remote_code,
            smoke_command=smoke_command,
        )
        if runtime_smoke and status == "ready"
        else None
    )
    if smoke_result is not None:
        status, reason = _status_from_smoke(smoke_result)
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
        runtime_smoke=runtime_smoke,
        runtime_timeout_seconds=runtime_timeout_seconds,
        trust_remote_code=trust_remote_code,
        smoke_result=smoke_result,
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
        runtime_smoke=runtime_smoke,
        runtime_timeout_seconds=runtime_timeout_seconds,
        trust_remote_code=trust_remote_code,
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
                nvidia_smi=nvidia_smi,
                generation_command=generation_command,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_seqs=max_num_seqs,
                runtime_smoke=runtime_smoke,
                runtime_timeout_seconds=runtime_timeout_seconds,
                trust_remote_code=trust_remote_code,
                smoke_result=smoke_result,
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
            runtime_smoke=runtime_smoke,
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
    runtime_smoke: bool,
    runtime_timeout_seconds: int,
    trust_remote_code: bool,
    smoke_result: SmokeResult | None,
) -> CacheProbeResult:
    metadata: dict[str, MetadataValue] = {
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
        "runtime_smoke_enabled": runtime_smoke,
        "runtime_timeout_seconds": runtime_timeout_seconds,
        "trust_remote_code": trust_remote_code,
        "runtime_smoke_status": (None if smoke_result is None else str(smoke_result["status"])),
        "runtime_smoke_returncode": (None if smoke_result is None else smoke_result["returncode"]),
        "runtime_smoke_command": (None if smoke_result is None else str(smoke_result["command"])),
        "runtime_smoke_stdout": (None if smoke_result is None else str(smoke_result["stdout"])),
        "runtime_smoke_stderr": (None if smoke_result is None else str(smoke_result["stderr"])),
        "cuda_available": environment.gpu.cuda_available,
        "nvidia_smi_status": nvidia_smi["status"],
        "nvidia_smi_returncode": nvidia_smi["returncode"],
        "nvidia_smi_stdout": nvidia_smi["stdout"],
        "nvidia_smi_stderr": nvidia_smi["stderr"],
        "measurement_kind": "vanilla-vllm-runtime-baseline",
        "python_executable": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "editable_install_used": False,
        "blocker_chain": _blocker_chain_text(
            vllm_installed=vllm_installed,
            cuda_available=environment.gpu.cuda_available,
            nvidia_smi_status=nvidia_smi["status"],
        ),
        "infrastructure_decision": INFRASTRUCTURE_DECISION,
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


def _run_runtime_smoke(
    *,
    model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    timeout_seconds: int,
    trust_remote_code: bool,
    smoke_command: str | None,
) -> SmokeResult:
    if timeout_seconds <= 0:
        raise ValueError("runtime_timeout_seconds must be positive")
    command = (
        shlex.split(smoke_command)
        if smoke_command is not None
        else _runtime_smoke_command(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            trust_remote_code=trust_remote_code,
        )
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "command": " ".join(shlex.quote(part) for part in command),
            "returncode": None,
            "stdout": _truncate_output(exc.stdout),
            "stderr": _truncate_output(exc.stderr),
            "timeout_seconds": timeout_seconds,
        }
    return {
        "status": "completed" if completed.returncode == 0 else "failed",
        "command": " ".join(shlex.quote(part) for part in command),
        "returncode": completed.returncode,
        "stdout": _truncate_output(completed.stdout),
        "stderr": _truncate_output(completed.stderr),
        "timeout_seconds": timeout_seconds,
    }


def _runtime_smoke_command(
    *,
    model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    trust_remote_code: bool,
) -> list[str]:
    code = "\n".join(
        [
            "from vllm import LLM",
            "llm = LLM(",
            f"    model={model!r},",
            f"    max_model_len={max_model_len!r},",
            f"    gpu_memory_utilization={gpu_memory_utilization!r},",
            f"    max_num_seqs={max_num_seqs!r},",
            f"    trust_remote_code={trust_remote_code!r},",
            ")",
            "print('vllm_model_load=ok')",
            "del llm",
        ]
    )
    return [sys.executable, "-c", code]


def _status_from_smoke(smoke_result: SmokeResult) -> tuple[str, str]:
    status = smoke_result["status"]
    if status == "completed":
        return "completed", "bounded vanilla vLLM model-load smoke completed"
    if status == "timeout":
        return "blocked", "bounded vanilla vLLM model-load smoke timed out"
    return "blocked", "bounded vanilla vLLM model-load smoke failed"


def _truncate_output(value: object, limit: int = 4000) -> str:
    if value is None:
        return ""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    return text[-limit:]


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
    nvidia_smi: dict[str, str | int | None],
    generation_command: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    runtime_smoke: bool,
    runtime_timeout_seconds: int,
    trust_remote_code: bool,
    smoke_result: SmokeResult | None,
) -> dict[str, object]:
    return {
        "artifact_name": "vllm-runtime-baseline",
        "generated_at": timestamp,
        "status": status,
        "reason": reason,
        "blocker_chain": _blocker_chain(
            vllm_installed=vllm_installed,
            cuda_available=environment.gpu.cuda_available,
            nvidia_smi_status=nvidia_smi["status"],
        ),
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
        "python_executable": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "editable_install_used": False,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "runtime_smoke_enabled": runtime_smoke,
        "runtime_timeout_seconds": runtime_timeout_seconds,
        "trust_remote_code": trust_remote_code,
        "runtime_smoke": smoke_result,
        "generation_command": generation_command,
        "infrastructure_decision": INFRASTRUCTURE_DECISION,
        "runtime_gate": {
            "nvidia_smi": "must exit 0 and report the target GPU",
            "torch_cuda": "torch.cuda.is_available() must be true with at least one device",
        },
    }


def _readme(
    *,
    generation_command: str,
    status: str,
    reason: str,
    model: str,
    fallback_model: str,
    runtime_smoke: bool,
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
- infrastructure decision: `{INFRASTRUCTURE_DECISION}`
- runtime smoke enabled: `{str(runtime_smoke).lower()}`

This step does not add vLLM to the main Cachepawl environment, run long-lived
serving, monkeypatch vLLM, replace allocators, add Triton kernels, add copy
kernels, add LSDR, or run real model quality evaluation.
"""


def _blocker_chain(
    *,
    vllm_installed: bool,
    cuda_available: bool,
    nvidia_smi_status: object,
) -> list[str]:
    blockers: list[str] = []
    if not vllm_installed:
        blockers.append("vllm is not installed in the active Python environment")
    if not cuda_available:
        blockers.append("torch reports CUDA unavailable")
    if nvidia_smi_status != "ok":
        blockers.append("nvidia-smi cannot initialize NVML successfully")
    return blockers


def _blocker_chain_text(
    *,
    vllm_installed: bool,
    cuda_available: bool,
    nvidia_smi_status: object,
) -> str:
    return " | ".join(
        _blocker_chain(
            vllm_installed=vllm_installed,
            cuda_available=cuda_available,
            nvidia_smi_status=nvidia_smi_status,
        )
    )


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
    runtime_smoke: bool,
    runtime_timeout_seconds: int,
    trust_remote_code: bool,
) -> str:
    prefix = "PYTHONPATH=src " if os.environ.get("PYTHONPATH") == "src" else ""
    return (
        f"{prefix}{sys.executable} benchmarks/scripts/capture_vllm_baseline.py "
        f"--output-dir {output_dir.as_posix()} "
        f"--model {json.dumps(model)} "
        f"--fallback-model {json.dumps(fallback_model)} "
        f"--timestamp {json.dumps(timestamp)} "
        f"--max-model-len {max_model_len} "
        f"--gpu-memory-utilization {gpu_memory_utilization} "
        f"--max-num-seqs {max_num_seqs} "
        f"--gpu-total-bytes {gpu_total_bytes} "
        f"--gpu-name {json.dumps(gpu_name)}"
        + (" --runtime-smoke" if runtime_smoke else "")
        + f" --runtime-timeout-seconds {runtime_timeout_seconds}"
        + (" --trust-remote-code" if trust_remote_code else "")
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
