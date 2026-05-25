#!/usr/bin/env python
"""Create advisory diagnostics from translated vLLM runtime cache output."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cachepawl.integrations.vllm import advise_vllm_runtime_cache_plan

DEFAULT_INPUT_DIR = Path("research/avmp/v2/results/vllm-runtime-cache-plan-observation")
DEFAULT_OUTPUT_DIR = Path("research/avmp/v2/results/vllm-runtime-cache-diagnostic")
JsonObject = dict[str, object]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=_timestamp())
    args = parser.parse_args(argv)

    create_diagnostic(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timestamp=args.timestamp,
    )
    return 0


def create_diagnostic(*, input_dir: Path, output_dir: Path, timestamp: str) -> JsonObject:
    translated_path = input_dir / "translated_runtime_cache_config.json"
    raw_metadata_path = input_dir / "raw_safe_metadata.json"
    translated = _read_json_object(translated_path)
    raw_metadata = _read_json_object(raw_metadata_path) if raw_metadata_path.exists() else {}

    report = advise_vllm_runtime_cache_plan(
        translated,
        raw_safe_metadata=raw_metadata,
    ).to_dict()
    manifest: JsonObject = {
        "artifact": "vllm-runtime-cache-diagnostic",
        "timestamp": timestamp,
        "status": report["primary_classification"],
        "source_translated_cache_config": str(translated_path),
        "source_raw_safe_metadata": str(raw_metadata_path) if raw_metadata_path.exists() else None,
        "observe_only": True,
        "vllm_behavior_changed": False,
        "scheduler_mutation": False,
        "allocator_replacement": False,
        "worker_layout_mutation": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(_summary(report, manifest))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "manifest": manifest,
        "report": report,
    }


def _summary(report: JsonObject, manifest: JsonObject) -> str:
    classifications = ", ".join(str(item) for item in _as_sequence(report["classifications"]))
    coverage = _as_json_object(report["planner_input_coverage"])
    missing_mutation = "\n".join(
        f"- {item}" for item in _as_sequence(report["missing_fields_for_mutation"])
    )
    return f"""# vLLM Runtime Cache Diagnostic

Status: `{manifest["status"]}`

This artifact is advisory and diagnostic only. It consumes Cachepawl's
translated runtime vLLM cache-plan observation and does not change vLLM
behavior. Scheduler decisions, allocator behavior, worker tensor layout,
serving behavior, Triton kernels, copy kernels, LSDR, and quality evaluation
remain unchanged.

## Classification

{classifications}

Cachepawl can recommend from observed planning metadata, but runtime improvement still requires
a future scheduler, allocator, planner, or worker allocation mutation point.

## Key Metrics

- `num_blocks`: {report["num_blocks"]}
- `cache_group_count`: {report["cache_group_count"]}
- `cache_tensor_count`: {report["cache_tensor_count"]}
- `layer_count`: {report["layer_count"]}
- `available_kv_cache_gpu_memory_bytes`: {report["available_kv_cache_gpu_memory_bytes"]}
- `observed_reserved_bytes`: {report["observed_reserved_bytes"]}
- `observed_useful_bytes`: {report["observed_useful_bytes"]}
- `cachepawl_recommended_bytes`: {report["cachepawl_recommended_bytes"]}
- `advisory_savings_bytes`: {report["advisory_savings_bytes"]}
- `overestimation_ratio`: {report["overestimation_ratio"]}
- `wasted_fraction`: {report["wasted_fraction"]}

## Planner Input Coverage

{_coverage_table(coverage)}

## Missing For Runtime Mutation

{missing_mutation}
"""


def _coverage_table(coverage: JsonObject) -> str:
    lines = ["| field | available |", "|---|---:|"]
    for key, value in sorted(coverage.items()):
        lines.append(f"| `{key}` | {str(value).lower()} |")
    return "\n".join(lines)


def _read_json_object(path: Path) -> JsonObject:
    data = json.loads(path.read_text())
    return _as_json_object(data)


def _as_json_object(value: object) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object, got {type(value).__name__}")
    return value


def _as_sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, str):
        raise TypeError("expected sequence, got string")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("expected sequence") from exc


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
