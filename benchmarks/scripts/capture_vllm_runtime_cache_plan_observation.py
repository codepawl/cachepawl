#!/usr/bin/env python
"""Capture a read-only runtime-resolved vLLM KVCacheConfig observation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from importlib.util import find_spec
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("research/avmp/v2/results/vllm-runtime-cache-plan-observation")
DEFAULT_MODEL = "Zyphra/Zamba2-2.7B-instruct"
PINNED_VLLM_VERSION = "0.21.0"
PINNED_VENV_PATH = "/tmp/vllm-cachepawl-venv"
OBSERVATION_PREFIX = "CACHEPAWL_RUNTIME_CACHE_PLAN="
JsonObject = dict[str, object]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=_timestamp())
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    capture_runtime_observation(
        output_dir=args.output_dir,
        timestamp=args.timestamp,
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        timeout_seconds=args.timeout_seconds,
        trust_remote_code=args.trust_remote_code,
    )
    return 0


def capture_runtime_observation(
    *,
    output_dir: Path,
    timestamp: str,
    model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    timeout_seconds: int,
    trust_remote_code: bool = False,
) -> JsonObject:
    output_dir.mkdir(parents=True, exist_ok=True)
    vllm_version = _optional_package_version("vllm")
    if find_spec("vllm") is None:
        blocker = _blocker(
            timestamp=timestamp,
            model=model,
            reason="vllm is not installed in the active Python environment",
            vllm_version=vllm_version,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            timeout_seconds=timeout_seconds,
            trust_remote_code=trust_remote_code,
        )
        _write_blocker(output_dir, blocker)
        return blocker

    cuda_available, device_count, device_name = _cuda_status()
    if not cuda_available:
        blocker = _blocker(
            timestamp=timestamp,
            model=model,
            reason="torch reports CUDA unavailable in the pinned vLLM environment",
            vllm_version=vllm_version,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            timeout_seconds=timeout_seconds,
            trust_remote_code=trust_remote_code,
            extra_metadata={
                "cuda_available": cuda_available,
                "device_count": device_count,
                "device_name": device_name,
            },
        )
        _write_blocker(output_dir, blocker)
        return blocker

    command = _runtime_observation_command(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        trust_remote_code=trust_remote_code,
    )
    result = _run_runtime_observation_child(command, timeout_seconds=timeout_seconds)
    if result["status"] != "completed":
        blocker = _blocker(
            timestamp=timestamp,
            model=model,
            reason=str(result["reason"]),
            vllm_version=vllm_version,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            timeout_seconds=timeout_seconds,
            trust_remote_code=trust_remote_code,
            extra_metadata=result,
        )
        _write_blocker(output_dir, blocker)
        return blocker

    payload = _success_payload(
        timestamp=timestamp,
        model=model,
        vllm_version=vllm_version,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        timeout_seconds=timeout_seconds,
        trust_remote_code=trust_remote_code,
        child_payload=_as_json_object(result["payload"]),
        child_metadata=result,
    )
    (output_dir / "translated_runtime_cache_config.json").write_text(
        json.dumps(payload["translated_runtime_cache_config"], indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "raw_safe_metadata.json").write_text(
        json.dumps(payload["raw_safe_metadata"], indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(payload["manifest"], indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "README.md").write_text(_readme_success(payload))
    blocker_path = output_dir / "blocker.json"
    if blocker_path.exists():
        blocker_path.unlink()
    return payload


def _run_runtime_observation_child(command: list[str], *, timeout_seconds: int) -> JsonObject:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env={**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "reason": "runtime vLLM observation timed out",
            "command": _format_command(command),
            "returncode": None,
            "stdout": _truncate_output(exc.stdout),
            "stderr": _truncate_output(exc.stderr),
        }

    payload = _parse_payload(completed.stdout)
    payload_status = payload.get("status") if payload is not None else None
    status = (
        "completed"
        if completed.returncode == 0 and payload_status == "runtime_resolved_translation"
        else "failed"
    )
    return {
        "status": status,
        "reason": (
            "runtime vLLM observation completed"
            if status == "completed"
            else _payload_failure_reason(payload)
        ),
        "command": _format_command(command),
        "returncode": completed.returncode,
        "stdout": _truncate_output(completed.stdout),
        "stderr": _truncate_output(completed.stderr),
        "payload": payload,
    }


def _runtime_observation_command(
    *,
    model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    trust_remote_code: bool,
) -> list[str]:
    code = "\n".join(
        [
            "import json",
            "from cachepawl.integrations.vllm import observe_vllm_runtime_cache_plan",
            "from vllm import LLM",
            "llm = LLM(",
            f"    model={model!r},",
            f"    max_model_len={max_model_len!r},",
            f"    gpu_memory_utilization={gpu_memory_utilization!r},",
            f"    max_num_seqs={max_num_seqs!r},",
            f"    trust_remote_code={trust_remote_code!r},",
            ")",
            "payload = observe_vllm_runtime_cache_plan(llm).to_dict()",
            f"print({OBSERVATION_PREFIX!r} + json.dumps(payload, sort_keys=True))",
            "del llm",
        ]
    )
    return [sys.executable, "-c", code]


def _payload_failure_reason(payload: JsonObject | None) -> str:
    if payload is None:
        return "runtime vLLM observation failed before emitting a payload"
    reason = payload.get("unsupported_reason")
    if reason is not None:
        return str(reason)
    status = payload.get("status")
    return f"runtime vLLM observation returned unsupported status {status!r}"


def _success_payload(
    *,
    timestamp: str,
    model: str,
    vllm_version: str | None,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    timeout_seconds: int,
    trust_remote_code: bool,
    child_payload: JsonObject,
    child_metadata: JsonObject,
) -> JsonObject:
    translated = _as_json_object(child_payload["translated_runtime_cache_config"])
    raw_safe_metadata = _as_json_object(child_payload["raw_safe_metadata"])
    manifest = {
        "artifact": "vllm-runtime-cache-plan-observation",
        "timestamp": timestamp,
        "status": "runtime_resolved_translation",
        "model": model,
        "pinned_vllm_version": PINNED_VLLM_VERSION,
        "vllm_version": vllm_version,
        "python_executable": sys.executable,
        "pinned_venv_path": PINNED_VENV_PATH,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "timeout_seconds": timeout_seconds,
        "trust_remote_code": trust_remote_code,
        "runtime_path": child_payload["runtime_path"],
        "manager_path_matches_scheduler": child_payload["manager_path_matches_scheduler"],
        "object_access": {
            "runtime_resolved_kv_cache_config": True,
            "model_loaded": True,
            "long_lived_serve": False,
            "allocator_replacement": False,
            "monkeypatching": False,
            "vllm_source_modified": False,
            "path_c_mutation": False,
        },
        "runtime_vs_direct_observation_comparison": _runtime_vs_direct_comparison(
            translated,
            raw_safe_metadata,
        ),
        "child_command": child_metadata["command"],
    }
    return {
        "manifest": manifest,
        "raw_safe_metadata": raw_safe_metadata,
        "translated_runtime_cache_config": translated,
    }


def _runtime_vs_direct_comparison(
    translated: JsonObject, raw_safe_metadata: JsonObject
) -> list[dict[str, str]]:
    group_count = translated.get("group_count")
    tensor_count = raw_safe_metadata.get("kv_cache_tensor_count")
    return [
        {
            "field": "runtime_path",
            "result": "new runtime-resolved object reached",
            "detail": "scheduler.kv_cache_config is available after vanilla LLM initialization",
        },
        {
            "field": "KVCacheConfig.num_blocks",
            "result": "runtime value",
            "detail": f"runtime planner resolved num_blocks={translated.get('num_blocks')}",
        },
        {
            "field": "KVCacheConfig.kv_cache_groups",
            "result": "runtime value",
            "detail": f"runtime planner resolved group_count={group_count}",
        },
        {
            "field": "KVCacheConfig.kv_cache_tensors",
            "result": "runtime value",
            "detail": f"runtime worker plan exposed tensor_count={tensor_count}",
        },
        {
            "field": "direct dataclass observation",
            "result": "translator assumptions still compatible",
            "detail": "same translator handled real runtime KVCacheConfig without vLLM imports",
        },
    ]


def _blocker(
    *,
    timestamp: str,
    model: str,
    reason: str,
    vllm_version: str | None,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    timeout_seconds: int,
    trust_remote_code: bool,
    extra_metadata: JsonObject | None = None,
) -> JsonObject:
    return {
        "manifest": {
            "artifact": "vllm-runtime-cache-plan-observation",
            "timestamp": timestamp,
            "status": "blocked",
            "reason": reason,
            "model": model,
            "pinned_vllm_version": PINNED_VLLM_VERSION,
            "vllm_version": vllm_version,
            "python_executable": sys.executable,
            "pinned_venv_path": PINNED_VENV_PATH,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "timeout_seconds": timeout_seconds,
            "trust_remote_code": trust_remote_code,
            "object_access": {
                "runtime_resolved_kv_cache_config": False,
                "long_lived_serve": False,
                "allocator_replacement": False,
                "monkeypatching": False,
                "vllm_source_modified": False,
                "path_c_mutation": False,
            },
            "metadata": extra_metadata or {},
        }
    }


def _write_blocker(output_dir: Path, blocker: JsonObject) -> None:
    (output_dir / "blocker.json").write_text(json.dumps(blocker, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(blocker["manifest"], indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "README.md").write_text(_readme_blocker(blocker))
    for stale in ("translated_runtime_cache_config.json", "raw_safe_metadata.json"):
        path = output_dir / stale
        if path.exists():
            path.unlink()


def _readme_success(payload: JsonObject) -> str:
    manifest = _as_json_object(payload["manifest"])
    comparison_items = _as_comparison_items(manifest["runtime_vs_direct_observation_comparison"])
    comparison = "\n".join(
        f"- `{item['field']}`: {item['result']} — {item['detail']}" for item in comparison_items
    )
    return f"""# vLLM Runtime Cache Plan Observation

Status: `{manifest["status"]}`

This artifact is a bounded, read-only runtime observation from vanilla
`vllm=={manifest["vllm_version"]}` in `{PINNED_VENV_PATH}` with `PYTHONPATH=src`.
It initializes an offline `LLM` for `{manifest["model"]}`, reads
`{manifest["runtime_path"]}`, and translates the resulting `KVCacheConfig` with
Cachepawl's import-safe translator.

No vLLM source edits, monkeypatching, allocator replacement, Path C mutation,
long-lived serving, Triton kernels, copy kernels, LSDR, or quality evaluation
were performed.

## Files

- `manifest.json` — capture status, path, parameters, and comparison.
- `translated_runtime_cache_config.json` — translated runtime planner output.
- `raw_safe_metadata.json` — scalar/list metadata only; no tensors or weights.

## Runtime vs Direct Observation

{comparison}

## Minimal Next Observe-First Step

Convert this bounded script into an observer helper that can be invoked around
vanilla engine initialization and persist the translated config alongside
baseline runs.
"""


def _readme_blocker(blocker: JsonObject) -> str:
    manifest = _as_json_object(blocker["manifest"])
    return f"""# vLLM Runtime Cache Plan Observation

Status: `blocked`

Reason: {manifest["reason"]}

No vLLM source edits, monkeypatching, allocator replacement, Path C mutation,
long-lived serving, Triton kernels, copy kernels, LSDR, or quality evaluation
were performed.
"""


def _parse_payload(stdout: str) -> JsonObject | None:
    for line in stdout.splitlines():
        if line.startswith(OBSERVATION_PREFIX):
            payload = json.loads(line.removeprefix(OBSERVATION_PREFIX))
            return _as_json_object(payload)
    return None


def _cuda_status() -> tuple[bool, int, str | None]:
    torch = importlib.import_module("torch")
    cuda = _attr(torch, "cuda")
    is_available = bool(_call(_attr(cuda, "is_available")))
    raw_device_count = _call(_attr(cuda, "device_count"))
    if not isinstance(raw_device_count, int):
        raise TypeError("torch.cuda.device_count() did not return an integer")
    device_count = raw_device_count
    device_name = str(_call(_attr(cuda, "get_device_name"), 0)) if is_available else None
    return is_available, device_count, device_name


def _optional_package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_json_object(value: object) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError(f"expected object, got {type(value).__name__}")
    return value


def _as_comparison_items(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError(f"expected comparison list, got {type(value).__name__}")
    items: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError(f"expected comparison item object, got {type(item).__name__}")
        items.append({str(key): str(val) for key, val in item.items()})
    return items


def _attr(obj: object, name: str) -> object:
    return getattr(obj, name)


def _call(func: object, *args: object) -> object:
    return func(*args)  # type: ignore[operator]


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _truncate_output(value: object, limit: int = 4000) -> str:
    if value is None:
        return ""
    text = value.decode() if isinstance(value, bytes) else str(value)
    if len(text) <= limit:
        return text
    return text[-limit:]


if __name__ == "__main__":
    raise SystemExit(main())
