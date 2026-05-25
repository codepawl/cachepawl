#!/usr/bin/env python
"""Capture a read-only planner-stage vLLM cache config observation."""

from __future__ import annotations

import argparse
import importlib
import inspect
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

DEFAULT_OUTPUT_DIR = Path("research/avmp/v2/results/vllm-planner-stage-observation")
DEFAULT_MODEL = "Zyphra/Zamba2-2.7B-instruct"
PINNED_VLLM_VERSION = "0.21.0"
PINNED_VENV_PATH = "/tmp/vllm-cachepawl-venv"
OBSERVATION_PREFIX = "CACHEPAWL_PLANNER_STAGE_OBSERVATION="
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

    capture_planner_stage_observation(
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


def capture_planner_stage_observation(
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
    planner_stage_metadata = _safe_planner_stage_metadata()
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
                "planner_stage_static_metadata": planner_stage_metadata,
            },
        )
        _write_blocker(output_dir, blocker)
        return blocker

    command = _planner_stage_observation_command(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        trust_remote_code=trust_remote_code,
    )
    result = _run_planner_stage_observation_child(command, timeout_seconds=timeout_seconds)
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
    (output_dir / "translated_planner_stage_config.json").write_text(
        json.dumps(payload["translated_planner_stage_config"], indent=2, sort_keys=True) + "\n"
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


def _run_planner_stage_observation_child(command: list[str], *, timeout_seconds: int) -> JsonObject:
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
            "reason": "planner-stage vLLM observation timed out",
            "command": _format_command(command),
            "returncode": None,
            "stdout": _truncate_output(exc.stdout),
            "stderr": _truncate_output(exc.stderr),
        }

    payload = _parse_payload(completed.stdout)
    payload_status = payload.get("status") if payload is not None else None
    status = (
        "completed"
        if completed.returncode == 0 and payload_status == "planner_stage_translation"
        else "failed"
    )
    return {
        "status": status,
        "reason": (
            "planner-stage vLLM observation completed"
            if status == "completed"
            else _payload_failure_reason(payload)
        ),
        "command": _format_command(command),
        "returncode": completed.returncode,
        "stdout": _truncate_output(completed.stdout),
        "stderr": _truncate_output(completed.stderr),
        "payload": payload,
    }


def _planner_stage_observation_command(
    *,
    model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    trust_remote_code: bool,
) -> list[str]:
    code = "\n".join(
        [
            "import copy",
            "import inspect",
            "import json",
            "from cachepawl.integrations.vllm import translate_kv_cache_config",
            "from vllm import LLM",
            "from vllm.v1.core.kv_cache_utils import get_kv_cache_configs",
            "llm = LLM(",
            f"    model={model!r},",
            f"    max_model_len={max_model_len!r},",
            f"    gpu_memory_utilization={gpu_memory_utilization!r},",
            f"    max_num_seqs={max_num_seqs!r},",
            f"    trust_remote_code={trust_remote_code!r},",
            ")",
            "core = llm.llm_engine.engine_core.engine_core",
            "vllm_config = core.vllm_config",
            "kv_cache_specs = core.model_executor.get_kv_cache_specs()",
            "available_scalar = int(core.available_gpu_memory_for_kv_cache)",
            "if len(kv_cache_specs) != 1:",
            "    raise RuntimeError(",
            "        'safe exact available-memory reconstruction is only implemented '",
            "        f'for one worker, got {len(kv_cache_specs)} workers'",
            "    )",
            "available_memory = [available_scalar]",
            "planner_configs = get_kv_cache_configs(",
            "    copy.deepcopy(vllm_config),",
            "    copy.deepcopy(kv_cache_specs),",
            "    list(available_memory),",
            ")",
            "runtime_config = core.scheduler.kv_cache_config",
            "translated = translate_kv_cache_config(planner_configs[0]).to_dict()",
            "runtime_translated = translate_kv_cache_config(runtime_config).to_dict()",
            "spec_type_counts = {}",
            "sample_layer_names = []",
            "for worker_specs in kv_cache_specs:",
            "    for layer_name, spec in worker_specs.items():",
            "        spec_type = type(spec).__name__",
            "        spec_type_counts[spec_type] = spec_type_counts.get(spec_type, 0) + 1",
            "        if len(sample_layer_names) < 12:",
            "            sample_layer_names.append(str(layer_name))",
            "payload = {",
            "    'status': 'planner_stage_translation',",
            "    'function_path': 'vllm.v1.core.kv_cache_utils.get_kv_cache_configs',",
            "    'input_paths': {",
            "        'vllm_config': 'LLM.llm_engine.engine_core.engine_core.vllm_config',",
            "        'kv_cache_specs': (",
            "            'LLM.llm_engine.engine_core.engine_core.model_executor.'",
            "            'get_kv_cache_specs()'",
            "        ),",
            "        'available_memory': (",
            "            'LLM.llm_engine.engine_core.engine_core.'",
            "            'available_gpu_memory_for_kv_cache'",
            "        ),",
            "    },",
            "    'observation_mode': 'post_init_replay_on_copied_real_inputs',",
            "    'get_kv_cache_configs_signature': str(inspect.signature(get_kv_cache_configs)),",
            "    'vllm_config_type': type(vllm_config).__name__,",
            "    'kv_cache_specs_worker_count': len(kv_cache_specs),",
            "    'kv_cache_specs_layer_counts': [len(specs) for specs in kv_cache_specs],",
            "    'kv_cache_spec_type_counts': spec_type_counts,",
            "    'sample_layer_names': sample_layer_names,",
            "    'available_memory': available_memory,",
            "    'planner_output_config_count': len(planner_configs),",
            "    'planner_output_num_blocks': [cfg.num_blocks for cfg in planner_configs],",
            "    'runtime_num_blocks': runtime_config.num_blocks,",
            "    'planner_matches_runtime_scheduler': translated == runtime_translated,",
            "    'translated_planner_stage_config': translated,",
            "}",
            f"print({OBSERVATION_PREFIX!r} + json.dumps(payload, sort_keys=True))",
            "del llm",
        ]
    )
    return [sys.executable, "-c", code]


def _payload_failure_reason(payload: JsonObject | None) -> str:
    if payload is None:
        return "planner-stage vLLM observation failed before emitting a payload"
    reason = payload.get("unsupported_reason")
    if reason is not None:
        return str(reason)
    status = payload.get("status")
    return f"planner-stage vLLM observation returned unsupported status {status!r}"


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
    translated = _as_json_object(child_payload["translated_planner_stage_config"])
    raw_safe_metadata: JsonObject = {
        "function_path": child_payload["function_path"],
        "input_paths": child_payload["input_paths"],
        "observation_mode": child_payload["observation_mode"],
        "get_kv_cache_configs_signature": child_payload["get_kv_cache_configs_signature"],
        "vllm_config_type": child_payload["vllm_config_type"],
        "kv_cache_specs_worker_count": child_payload["kv_cache_specs_worker_count"],
        "kv_cache_specs_layer_counts": child_payload["kv_cache_specs_layer_counts"],
        "kv_cache_spec_type_counts": child_payload["kv_cache_spec_type_counts"],
        "sample_layer_names": child_payload["sample_layer_names"],
        "available_memory": child_payload["available_memory"],
        "planner_output_config_count": child_payload["planner_output_config_count"],
        "planner_output_num_blocks": child_payload["planner_output_num_blocks"],
        "runtime_num_blocks": child_payload["runtime_num_blocks"],
        "planner_matches_runtime_scheduler": child_payload["planner_matches_runtime_scheduler"],
    }
    manifest: JsonObject = {
        "artifact": "vllm-planner-stage-observation",
        "timestamp": timestamp,
        "status": "planner_stage_translation",
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
        "function_path": child_payload["function_path"],
        "observation_mode": child_payload["observation_mode"],
        "object_access": {
            "vllm_config": True,
            "kv_cache_specs": True,
            "available_memory": True,
            "get_kv_cache_configs_called": True,
            "copied_real_inputs": True,
            "returned_to_vllm": False,
            "long_lived_serve": False,
            "allocator_replacement": False,
            "monkeypatching": False,
            "vllm_source_modified": False,
            "scheduler_mutation": False,
            "worker_layout_mutation": False,
        },
        "input_availability": {
            "vllm_config": True,
            "kv_cache_specs": True,
            "available_memory": True,
            "available_memory_exact_for_worker_count": (
                child_payload["kv_cache_specs_worker_count"] == 1
            ),
        },
        "child_command": child_metadata["command"],
    }
    return {
        "manifest": manifest,
        "raw_safe_metadata": raw_safe_metadata,
        "translated_planner_stage_config": translated,
    }


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
            "artifact": "vllm-planner-stage-observation",
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
            "function_path": "vllm.v1.core.kv_cache_utils.get_kv_cache_configs",
            "object_access": {
                "vllm_config": False,
                "kv_cache_specs": False,
                "available_memory": False,
                "get_kv_cache_configs_called": False,
                "returned_to_vllm": False,
                "long_lived_serve": False,
                "allocator_replacement": False,
                "monkeypatching": False,
                "vllm_source_modified": False,
                "scheduler_mutation": False,
                "worker_layout_mutation": False,
            },
            "input_availability": {
                "vllm_config": False,
                "kv_cache_specs": False,
                "available_memory": False,
            },
            "metadata": extra_metadata or {},
        }
    }


def _write_blocker(output_dir: Path, blocker: JsonObject) -> None:
    (output_dir / "blocker.json").write_text(json.dumps(blocker, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(blocker["manifest"], indent=2, sort_keys=True) + "\n"
    )
    manifest = _as_json_object(blocker["manifest"])
    metadata = _as_json_object(manifest["metadata"])
    raw_metadata_path = output_dir / "raw_safe_metadata.json"
    if metadata:
        raw_metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    elif raw_metadata_path.exists():
        raw_metadata_path.unlink()
    (output_dir / "README.md").write_text(_readme_blocker(blocker))
    for stale in ("translated_planner_stage_config.json",):
        path = output_dir / stale
        if path.exists():
            path.unlink()


def _readme_success(payload: JsonObject) -> str:
    manifest = _as_json_object(payload["manifest"])
    metadata = _as_json_object(payload["raw_safe_metadata"])
    return f"""# vLLM Planner-Stage Observation

Status: `{manifest["status"]}`

This artifact records a bounded, read-only planner-stage observation from
vanilla `vllm=={manifest["vllm_version"]}` in `{PINNED_VENV_PATH}` with
`PYTHONPATH=src`. It reaches `{manifest["function_path"]}` by replaying the
planner on deep-copied real inputs observed after vanilla `LLM` initialization.

The computed Cachepawl translation is not returned to vLLM. No vLLM source
edits, monkeypatching, allocator replacement, scheduler mutation, worker layout
mutation, long-lived serving, Triton kernels, copy kernels, LSDR, or quality
evaluation were performed.

## Files

- `manifest.json` — capture status, parameters, non-mutation flags, and paths.
- `translated_planner_stage_config.json` — translated planner-stage output.
- `raw_safe_metadata.json` — scalar/list metadata only; no tensors or weights.

## Real Input Availability

- `VllmConfig`: available from `{_input_path(metadata, "vllm_config")}`.
- `KVCacheSpec` maps: available from `{_input_path(metadata, "kv_cache_specs")}`.
- Available memory: available from `{_input_path(metadata, "available_memory")}`.

## Result

- `observation_mode`: `{metadata["observation_mode"]}`
- `kv_cache_specs_worker_count`: {metadata["kv_cache_specs_worker_count"]}
- `kv_cache_specs_layer_counts`: {metadata["kv_cache_specs_layer_counts"]}
- `available_memory`: {metadata["available_memory"]}
- `planner_output_num_blocks`: {metadata["planner_output_num_blocks"]}
- `runtime_num_blocks`: {metadata["runtime_num_blocks"]}
- `planner_matches_runtime_scheduler`: {metadata["planner_matches_runtime_scheduler"]}

## Minimal Next Step

Use this same copied-input planner-stage replay as the source for a dry-run
comparison against Cachepawl recommendations. Runtime behavior remains unchanged
until a later explicit mutation decision identifies a safe scheduler or
allocator control point.
"""


def _readme_blocker(blocker: JsonObject) -> str:
    manifest = _as_json_object(blocker["manifest"])
    return f"""# vLLM Planner-Stage Observation

Status: `blocked`

Reason: {manifest["reason"]}

This artifact attempted to reach
`vllm.v1.core.kv_cache_utils.get_kv_cache_configs(...)` without mutating vLLM.
No vLLM source edits, monkeypatching, returned plans, allocator replacement,
scheduler mutation, worker layout mutation, long-lived serving, Triton kernels,
copy kernels, LSDR, or quality evaluation were performed.
"""


def _input_path(metadata: JsonObject, key: str) -> object:
    input_paths = _as_json_object(metadata["input_paths"])
    return input_paths[key]


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


def _safe_planner_stage_metadata() -> JsonObject:
    try:
        kv_cache_utils = importlib.import_module("vllm.v1.core.kv_cache_utils")
        engine_core = importlib.import_module("vllm.v1.engine.core")
        get_kv_cache_configs = _attr(kv_cache_utils, "get_kv_cache_configs")
        initialize_kv_caches = _attr(_attr(engine_core, "EngineCore"), "_initialize_kv_caches")
        return {
            "function_path": "vllm.v1.core.kv_cache_utils.get_kv_cache_configs",
            "function_signature": str(
                inspect.signature(get_kv_cache_configs)  # type: ignore[arg-type]
            ),
            "call_site_path": "vllm.v1.engine.core.EngineCore._initialize_kv_caches",
            "call_site_signature": str(
                inspect.signature(initialize_kv_caches)  # type: ignore[arg-type]
            ),
            "observed_input_paths_from_source": {
                "vllm_config": "EngineCore._initialize_kv_caches(vllm_config)",
                "kv_cache_specs": "self.model_executor.get_kv_cache_specs()",
                "available_memory": "self.model_executor.determine_available_memory()",
                "stored_available_memory": "self.available_gpu_memory_for_kv_cache",
            },
        }
    except Exception as exc:  # pragma: no cover - depends on optional vLLM import state.
        return {
            "metadata_error": f"{type(exc).__name__}: {exc}",
        }


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
