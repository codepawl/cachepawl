#!/usr/bin/env python
"""Capture a bounded read-only live-request observation from vanilla vLLM."""

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

DEFAULT_OUTPUT_DIR = Path("research/avmp/v2/results/vllm-live-request-contract-observation")
DEFAULT_MODEL = "Zyphra/Zamba2-2.7B-instruct"
DEFAULT_PROMPT = "Count from one to four:"
PINNED_VLLM_VERSION = "0.21.0"
PINNED_VENV_PATH = "/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv"
OBSERVATION_PREFIX = "CACHEPAWL_LIVE_REQUEST_CONTRACT_OBSERVATION="
JsonObject = dict[str, object]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=_timestamp())
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    capture_live_request_contract_observation(
        output_dir=args.output_dir,
        timestamp=args.timestamp,
        model=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        timeout_seconds=args.timeout_seconds,
        trust_remote_code=args.trust_remote_code,
    )
    return 0


def capture_live_request_contract_observation(
    *,
    output_dir: Path,
    timestamp: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    timeout_seconds: int,
    trust_remote_code: bool = False,
) -> JsonObject:
    _validate_bounds(
        max_new_tokens=max_new_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    vllm_version = _optional_package_version("vllm")
    if find_spec("vllm") is None:
        blocker = _blocker(
            timestamp=timestamp,
            model=model,
            prompt=prompt,
            reason="vllm is not installed in the active Python environment",
            vllm_version=vllm_version,
            max_new_tokens=max_new_tokens,
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
            prompt=prompt,
            reason="torch reports CUDA unavailable in the pinned vLLM environment",
            vllm_version=vllm_version,
            max_new_tokens=max_new_tokens,
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

    command = _live_request_contract_command(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        trust_remote_code=trust_remote_code,
    )
    result = _run_live_request_child(command, timeout_seconds=timeout_seconds)
    if result["status"] != "completed":
        blocker = _blocker(
            timestamp=timestamp,
            model=model,
            prompt=prompt,
            reason=str(result["reason"]),
            vllm_version=vllm_version,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            timeout_seconds=timeout_seconds,
            trust_remote_code=trust_remote_code,
            extra_metadata=result,
        )
        _write_blocker(output_dir, blocker)
        return blocker

    payload = create_live_request_contract_artifact_payload(
        timestamp=timestamp,
        model=model,
        prompt=prompt,
        vllm_version=vllm_version,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        timeout_seconds=timeout_seconds,
        trust_remote_code=trust_remote_code,
        child_payload=_as_json_object(result["payload"]),
        child_metadata=result,
    )
    write_live_request_contract_artifact(output_dir, payload)
    return payload


def create_live_request_contract_artifact_payload(
    *,
    timestamp: str,
    model: str,
    prompt: str,
    vllm_version: str | None,
    max_new_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    timeout_seconds: int,
    trust_remote_code: bool,
    child_payload: JsonObject,
    child_metadata: JsonObject,
) -> JsonObject:
    fields = _as_json_array(child_payload["fields"])
    blockers = _as_json_array(child_payload["field_level_blockers"])
    snapshots = _as_json_array(child_payload["snapshots"])
    report: JsonObject = {
        "status": "live_request_contract_observation_available",
        "classification": _classification(blockers),
        "model": model,
        "runtime_path": child_payload["runtime_path"],
        "request_id": child_payload["request_id"],
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "snapshots": snapshots,
        "output_metadata": _as_json_object(child_payload["output_metadata"]),
        "fields": fields,
        "field_level_blockers": blockers,
        "object_access": _as_json_object(child_payload["object_access"]),
        "non_mutating": True,
        "returned_to_vllm": False,
        "vllm_behavior_changed": False,
    }
    raw_safe_metadata = _as_json_object(child_payload["raw_safe_metadata"])
    manifest: JsonObject = {
        "artifact": "vllm-live-request-contract-observation",
        "timestamp": timestamp,
        "status": report["status"],
        "classification": report["classification"],
        "model": model,
        "pinned_vllm_version": PINNED_VLLM_VERSION,
        "vllm_version": vllm_version,
        "python_executable": sys.executable,
        "pinned_venv_path": PINNED_VENV_PATH,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "timeout_seconds": timeout_seconds,
        "trust_remote_code": trust_remote_code,
        "field_blocker_count": len(blockers),
        "outputs": {
            "readme": "README.md",
            "manifest": "manifest.json",
            "live_request_contract_report": "live_request_contract_report.json",
            "summary": "summary.md",
            "raw_safe_metadata": "raw_safe_metadata.json",
            "field_level_blockers": "field_level_blockers.json",
        },
        "object_access": report["object_access"],
        "child_command": child_metadata["command"],
    }
    return {
        "manifest": manifest,
        "live_request_contract_report": report,
        "raw_safe_metadata": raw_safe_metadata,
        "field_level_blockers": blockers,
    }


def write_live_request_contract_artifact(output_dir: Path, payload: JsonObject) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(_json_dumps(_as_json_object(payload["manifest"])))
    (output_dir / "live_request_contract_report.json").write_text(
        _json_dumps(_as_json_object(payload["live_request_contract_report"]))
    )
    (output_dir / "raw_safe_metadata.json").write_text(
        _json_dumps(_as_json_object(payload["raw_safe_metadata"]))
    )
    (output_dir / "field_level_blockers.json").write_text(
        _json_dumps({"field_level_blockers": payload["field_level_blockers"]})
    )
    (output_dir / "summary.md").write_text(_summary(payload))
    (output_dir / "README.md").write_text(_readme(payload))
    blocker_path = output_dir / "blocker.json"
    if blocker_path.exists():
        blocker_path.unlink()


def _run_live_request_child(command: list[str], *, timeout_seconds: int) -> JsonObject:
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
            "reason": "live request contract observation timed out",
            "command": _format_command(command),
            "returncode": None,
            "stdout": _truncate_output(exc.stdout),
            "stderr": _truncate_output(exc.stderr),
        }

    payload = _parse_payload(completed.stdout)
    payload_status = payload.get("status") if payload is not None else None
    status = (
        "completed"
        if completed.returncode == 0 and payload_status == "live_request_contract_observation"
        else "failed"
    )
    return {
        "status": status,
        "reason": (
            "live request contract observation completed"
            if status == "completed"
            else _payload_failure_reason(payload)
        ),
        "command": _format_command(command),
        "returncode": completed.returncode,
        "stdout": _truncate_output(completed.stdout),
        "stderr": _truncate_output(completed.stderr),
        "payload": payload,
    }


def _live_request_contract_command(
    *,
    model: str,
    prompt: str,
    max_new_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    trust_remote_code: bool,
) -> list[str]:
    code = "\n".join(
        [
            "import json",
            "from cachepawl.integrations.vllm import observe_vllm_live_request_contract",
            "from vllm import LLM, SamplingParams",
            "llm = LLM(",
            f"    model={model!r},",
            f"    max_model_len={max_model_len!r},",
            f"    gpu_memory_utilization={gpu_memory_utilization!r},",
            f"    max_num_seqs={max_num_seqs!r},",
            f"    trust_remote_code={trust_remote_code!r},",
            ")",
            "sampling_params = SamplingParams(",
            f"    max_tokens={max_new_tokens!r},",
            "    temperature=0.0,",
            "    ignore_eos=True,",
            ")",
            "payload = observe_vllm_live_request_contract(",
            "    llm,",
            f"    prompt={prompt!r},",
            "    sampling_params=sampling_params,",
            f"    max_new_tokens={max_new_tokens!r},",
            ").to_dict()",
            f"print({OBSERVATION_PREFIX!r} + json.dumps(payload, sort_keys=True))",
            "del llm",
        ]
    )
    return [sys.executable, "-c", code]


def _payload_failure_reason(payload: JsonObject | None) -> str:
    if payload is None:
        return "live request contract observation failed before emitting a payload"
    reason = payload.get("unsupported_reason")
    if reason is not None:
        return str(reason)
    status = payload.get("status")
    return f"live request contract observation returned unsupported status {status!r}"


def _classification(blockers: list[object]) -> str:
    return (
        "live_request_contract_observation_with_field_blockers"
        if blockers
        else "live_request_contract_observation_complete"
    )


def _blocker(
    *,
    timestamp: str,
    model: str,
    prompt: str,
    reason: str,
    vllm_version: str | None,
    max_new_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    timeout_seconds: int,
    trust_remote_code: bool,
    extra_metadata: JsonObject | None = None,
) -> JsonObject:
    return {
        "manifest": {
            "artifact": "vllm-live-request-contract-observation",
            "timestamp": timestamp,
            "status": "blocked",
            "reason": reason,
            "model": model,
            "prompt": prompt,
            "pinned_vllm_version": PINNED_VLLM_VERSION,
            "vllm_version": vllm_version,
            "python_executable": sys.executable,
            "pinned_venv_path": PINNED_VENV_PATH,
            "max_new_tokens": max_new_tokens,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "timeout_seconds": timeout_seconds,
            "trust_remote_code": trust_remote_code,
            "object_access": {
                "runtime_contract_objects_reached": False,
                "live_request_scheduled": False,
                "long_lived_serve": False,
                "allocator_replacement": False,
                "monkeypatching": False,
                "vllm_source_modified": False,
                "scheduler_mutation": False,
                "worker_layout_mutation": False,
                "returned_to_vllm": False,
                "controlled_substitution": False,
            },
            "metadata": extra_metadata or {},
        }
    }


def _write_blocker(output_dir: Path, blocker: JsonObject) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "blocker.json").write_text(_json_dumps(blocker))
    (output_dir / "manifest.json").write_text(_json_dumps(_as_json_object(blocker["manifest"])))
    (output_dir / "README.md").write_text(_readme_blocker(blocker))
    for stale in (
        "live_request_contract_report.json",
        "raw_safe_metadata.json",
        "field_level_blockers.json",
        "summary.md",
    ):
        path = output_dir / stale
        if path.exists():
            path.unlink()


def _summary(payload: JsonObject) -> str:
    manifest = _as_json_object(payload["manifest"])
    report = _as_json_object(payload["live_request_contract_report"])
    fields = _as_json_array(report["fields"])
    blockers = _as_json_array(report["field_level_blockers"])
    observed_lines = "\n".join(
        f"- `{_field_name(field)}`: `{_field_status(field)}`" for field in fields
    )
    blocker_lines = (
        "\n".join(f"- `{_field_name(field)}`: {_field_reason(field)}" for field in blockers)
        or "- none"
    )
    return f"""# vLLM Live Request Contract Observation Summary

Status: `{manifest["classification"]}`

Request id: `{report["request_id"]}`

This artifact is a bounded, read-only live-request observation from vanilla
`vllm=={manifest["vllm_version"]}` in `{PINNED_VENV_PATH}` with `PYTHONPATH=src`.

No vLLM source edits, monkeypatching, allocator replacement, scheduler
mutation, worker layout mutation, returned Cachepawl plans, controlled
substitution, Triton kernels, copy kernels, LSDR, serving changes, or quality
evaluation were performed.

## Fields

{observed_lines}

## Field-Level Blockers

{blocker_lines}
"""


def _readme(payload: JsonObject) -> str:
    manifest = _as_json_object(payload["manifest"])
    return f"""# vLLM Live Request Contract Observation

Status: `{manifest["classification"]}`

This artifact captures bounded read-only live-request metadata from vanilla
`vllm=={manifest["vllm_version"]}` for `{manifest["model"]}`.

Files:

- `manifest.json` - capture status, parameters, outputs, and non-mutation flags.
- `live_request_contract_report.json` - live request id, request/block snapshots,
  scheduler request metadata, and field status records.
- `raw_safe_metadata.json` - scalar runtime object metadata only.
- `field_level_blockers.json` - fields not safely observable in this bounded run.
- `summary.md` - concise human-readable summary.
"""


def _readme_blocker(blocker: JsonObject) -> str:
    manifest = _as_json_object(blocker["manifest"])
    return f"""# vLLM Live Request Contract Observation

Status: `blocked`

Reason: {manifest["reason"]}

No vLLM source edits, monkeypatching, allocator replacement, scheduler
mutation, worker layout mutation, returned Cachepawl plans, controlled
substitution, Triton kernels, copy kernels, LSDR, serving changes, or quality
evaluation were performed.
"""


def _validate_bounds(
    *,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
) -> None:
    if max_new_tokens < 1 or max_new_tokens > 8:
        raise ValueError("max_new_tokens must be between 1 and 8")
    if gpu_memory_utilization > 0.7:
        raise ValueError("gpu_memory_utilization must be no higher than 0.7")
    if max_num_seqs != 1:
        raise ValueError("max_num_seqs must be 1")


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
    device_name = str(_call(_attr(cuda, "get_device_name"), 0)) if is_available else None
    return is_available, raw_device_count, device_name


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


def _as_json_array(value: object) -> list[object]:
    if isinstance(value, str) or not isinstance(value, list):
        raise TypeError(f"expected array, got {type(value).__name__}")
    return value


def _field_name(field: object) -> str:
    return str(_as_json_object(field)["name"])


def _field_status(field: object) -> str:
    return str(_as_json_object(field)["status"])


def _field_reason(field: object) -> str:
    reason = _as_json_object(field).get("blocker_reason")
    return str(reason) if reason is not None else "blocked"


def _json_dumps(value: JsonObject) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


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
