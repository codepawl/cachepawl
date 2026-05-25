#!/usr/bin/env python
"""Capture a read-only vLLM cache-plan translation observation."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from importlib.util import find_spec
from inspect import signature
from pathlib import Path
from typing import Protocol, cast

from cachepawl.integrations.vllm import translate_kv_cache_config

DEFAULT_OUTPUT_DIR = Path("research/avmp/v2/results/vllm-cache-plan-observation")
PINNED_VLLM_VERSION = "0.21.0"
PINNED_VENV_PATH = "/tmp/vllm-cachepawl-venv"
JsonObject = dict[str, object]


class ObjectFactory(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=_timestamp())
    args = parser.parse_args(argv)

    capture_observation(output_dir=args.output_dir, timestamp=args.timestamp)
    return 0


def capture_observation(*, output_dir: Path, timestamp: str) -> JsonObject:
    output_dir.mkdir(parents=True, exist_ok=True)
    vllm_version = _optional_package_version("vllm")
    if find_spec("vllm") is None:
        blocker = _blocker(
            timestamp=timestamp,
            reason="vllm is not installed in the active Python environment",
            vllm_version=vllm_version,
        )
        _write_blocker(output_dir, blocker)
        return blocker

    try:
        payload = _capture_real_object_translation(
            timestamp=timestamp,
            vllm_version=vllm_version,
        )
    except Exception as exc:  # pragma: no cover - exercised only in pinned vLLM env.
        blocker = _blocker(
            timestamp=timestamp,
            reason=f"{type(exc).__name__}: {exc}",
            vllm_version=vllm_version,
        )
        _write_blocker(output_dir, blocker)
        return blocker

    (output_dir / "translated_cache_config.json").write_text(
        json.dumps(payload["translated_cache_config"], indent=2, sort_keys=True) + "\n"
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


def _capture_real_object_translation(*, timestamp: str, vllm_version: str | None) -> JsonObject:
    torch = importlib.import_module("torch")
    kv_cache_utils = importlib.import_module("vllm.v1.core.kv_cache_utils")
    kv_cache_interface = importlib.import_module("vllm.v1.kv_cache_interface")
    attention_spec_cls = _class_from_module(kv_cache_interface, "AttentionSpec")
    kv_cache_config_cls = _class_from_module(kv_cache_interface, "KVCacheConfig")
    kv_cache_group_spec_cls = _class_from_module(kv_cache_interface, "KVCacheGroupSpec")
    kv_cache_tensor_cls = _class_from_module(kv_cache_interface, "KVCacheTensor")
    mamba_spec_cls = _class_from_module(kv_cache_interface, "MambaSpec")
    torch_bfloat16 = _attr(torch, "bfloat16")

    attention_spec = attention_spec_cls(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch_bfloat16,
    )
    mamba_spec = mamba_spec_cls(
        block_size=1,
        shapes=((8192, 16),),
        dtypes=(torch_bfloat16,),
        mamba_cache_mode="align",
    )
    cache_config = kv_cache_config_cls(
        num_blocks=32,
        kv_cache_tensors=[
            kv_cache_tensor_cls(size=2_097_152, shared_by=["layers.0.self_attn"]),
            kv_cache_tensor_cls(size=8_388_608, shared_by=["layers.1.mamba"]),
        ],
        kv_cache_groups=[
            kv_cache_group_spec_cls(
                layer_names=["layers.0.self_attn", "layers.8.self_attn"],
                kv_cache_spec=attention_spec,
            ),
            kv_cache_group_spec_cls(
                layer_names=["layers.1.mamba"],
                kv_cache_spec=mamba_spec,
            ),
        ],
    )
    translated = translate_kv_cache_config(cache_config).to_dict()
    cache_groups = cast(list[object], _attr(cache_config, "kv_cache_groups"))
    cache_tensors = cast(list[object], _attr(cache_config, "kv_cache_tensors"))
    raw_metadata = {
        "classes": {
            "AttentionSpec": _class_metadata(attention_spec_cls),
            "MambaSpec": _class_metadata(mamba_spec_cls),
            "KVCacheGroupSpec": _class_metadata(kv_cache_group_spec_cls),
            "KVCacheTensor": _class_metadata(kv_cache_tensor_cls),
            "KVCacheConfig": _class_metadata(kv_cache_config_cls),
        },
        "get_kv_cache_configs_signature": str(
            signature(cast(ObjectFactory, _attr(kv_cache_utils, "get_kv_cache_configs")))
        ),
        "direct_object_values": {
            "attention_page_size_bytes": _attr(attention_spec, "page_size_bytes"),
            "attention_real_page_size_bytes": _attr(attention_spec, "real_page_size_bytes"),
            "mamba_page_size_bytes": _attr(mamba_spec, "page_size_bytes"),
            "mamba_shapes_type": type(_attr(mamba_spec, "shapes")).__name__,
            "mamba_dtypes_type": type(_attr(mamba_spec, "dtypes")).__name__,
            "kv_cache_group_layer_names_type": type(_attr(cache_groups[0], "layer_names")).__name__,
            "kv_cache_tensor_shared_by_type": type(_attr(cache_tensors[0], "shared_by")).__name__,
        },
    }
    manifest = {
        "artifact": "vllm-cache-plan-observation",
        "timestamp": timestamp,
        "status": "direct_real_object_translation",
        "pinned_vllm_version": PINNED_VLLM_VERSION,
        "vllm_version": vllm_version,
        "python_executable": sys.executable,
        "pinned_venv_path": PINNED_VENV_PATH,
        "object_access": {
            "direct_real_vllm_objects": True,
            "get_kv_cache_configs_called": False,
            "get_kv_cache_configs_skip_reason": (
                "requires VllmConfig, per-worker KVCacheSpec dictionaries, and "
                "available-memory inputs; runtime-resolved config capture remains "
                "the next observe-first step"
            ),
            "model_loaded": False,
            "allocator_replacement": False,
            "monkeypatching": False,
            "vllm_source_modified": False,
        },
        "fake_vs_real_assumption_comparison": _fake_vs_real_comparison(),
    }
    return {
        "manifest": manifest,
        "raw_safe_metadata": raw_metadata,
        "translated_cache_config": translated,
    }


def _class_from_module(module: object, name: str) -> ObjectFactory:
    return cast(ObjectFactory, _attr(module, name))


def _class_metadata(cls: ObjectFactory) -> JsonObject:
    dataclass_fields = (
        _attr(cls, "__dataclass_fields__") if hasattr(cls, "__dataclass_fields__") else None
    )
    metadata: JsonObject = {
        "signature": str(signature(cls)),
        "is_dataclass": dataclass_fields is not None,
    }
    if isinstance(dataclass_fields, dict):
        metadata["fields"] = sorted(str(name) for name in dataclass_fields)
    return metadata


def _fake_vs_real_comparison() -> list[dict[str, str]]:
    return [
        {
            "field": "AttentionSpec.page_size_bytes",
            "result": "matches",
            "detail": "real vLLM exposes page_size_bytes as an observable property",
        },
        {
            "field": "AttentionSpec.dtype",
            "result": "compatible",
            "detail": "real vLLM uses torch.dtype; translator stringifies it",
        },
        {
            "field": "MambaSpec.shapes",
            "result": "fake assumption widened",
            "detail": "fake tests used dicts; real vLLM uses tuple[tuple[int, ...], ...]",
        },
        {
            "field": "MambaSpec.dtypes",
            "result": "fake assumption widened",
            "detail": "fake tests used dicts; real vLLM uses tuple[torch.dtype, ...]",
        },
        {
            "field": "KVCacheGroupSpec.layer_names",
            "result": "compatible",
            "detail": "real vLLM uses lists; translator normalizes to tuples",
        },
        {
            "field": "KVCacheConfig.block_size/cache_dtype",
            "result": "not present",
            "detail": "real vLLM 0.21.0 KVCacheConfig only has num_blocks, tensors, and groups",
        },
    ]


def _blocker(*, timestamp: str, reason: str, vllm_version: str | None) -> JsonObject:
    return {
        "manifest": {
            "artifact": "vllm-cache-plan-observation",
            "timestamp": timestamp,
            "status": "blocked",
            "reason": reason,
            "pinned_vllm_version": PINNED_VLLM_VERSION,
            "vllm_version": vllm_version,
            "python_executable": sys.executable,
            "pinned_venv_path": PINNED_VENV_PATH,
            "allocator_replacement": False,
            "monkeypatching": False,
            "vllm_source_modified": False,
        }
    }


def _write_blocker(output_dir: Path, blocker: JsonObject) -> None:
    (output_dir / "blocker.json").write_text(json.dumps(blocker, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(blocker["manifest"], indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "README.md").write_text(_readme_blocker(blocker))
    for stale in ("translated_cache_config.json", "raw_safe_metadata.json"):
        path = output_dir / stale
        if path.exists():
            path.unlink()


def _readme_success(payload: JsonObject) -> str:
    manifest = cast(JsonObject, payload["manifest"])
    comparison_items = cast(list[dict[str, str]], manifest["fake_vs_real_assumption_comparison"])
    comparison = "\n".join(
        f"- `{item['field']}`: {item['result']} — {item['detail']}" for item in comparison_items
    )
    return f"""# vLLM Cache Plan Observation

Status: `{manifest["status"]}`

This artifact is a read-only direct real vLLM object translation captured from
`vllm=={manifest["vllm_version"]}` in `{PINNED_VENV_PATH}` with `PYTHONPATH=src`.
It constructs vLLM 0.21.0 cache planning dataclasses and translates them with
Cachepawl's import-safe translator. It does not load a model, call
`get_kv_cache_configs`, modify vLLM source, monkeypatch vLLM, replace
allocators, or implement Path C mutation.

## Files

- `manifest.json` — capture status, scope, and fake-vs-real comparison.
- `translated_cache_config.json` — Cachepawl-owned translated snapshot.
- `raw_safe_metadata.json` — signatures and scalar metadata only; no tensors.

## Fake-vs-Real Assumption Comparison

{comparison}

## Minimal Next Observe-First Step

Capture a runtime-resolved `KVCacheConfig` from a vanilla vLLM engine or worker
after planning, then run the same translator against that object.
"""


def _readme_blocker(blocker: JsonObject) -> str:
    manifest = cast(JsonObject, blocker["manifest"])
    return f"""# vLLM Cache Plan Observation

Status: `blocked`

Reason: {manifest["reason"]}

No vLLM source edits, monkeypatching, allocator replacement, model loading, or
Path C mutation were performed.
"""


def _optional_package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _attr(obj: object, name: str) -> object:
    return getattr(obj, name)


if __name__ == "__main__":
    raise SystemExit(main())
