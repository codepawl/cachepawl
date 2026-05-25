#!/usr/bin/env python
"""Create a non-mutating planner dry-run probe from vLLM runtime cache output."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cachepawl.integrations.vllm import dry_run_vllm_planner_probe

DEFAULT_INPUT_DIR = Path("research/avmp/v2/results/vllm-runtime-cache-plan-observation")
DEFAULT_OUTPUT_DIR = Path("research/avmp/v2/results/vllm-planner-dry-run-probe")
JsonObject = dict[str, object]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=_timestamp())
    args = parser.parse_args(argv)

    create_planner_dry_run_probe(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timestamp=args.timestamp,
    )
    return 0


def create_planner_dry_run_probe(
    *,
    input_dir: Path,
    output_dir: Path,
    timestamp: str,
) -> JsonObject:
    translated_path = input_dir / "translated_runtime_cache_config.json"
    raw_metadata_path = input_dir / "raw_safe_metadata.json"
    translated = _read_json_object(translated_path)
    raw_metadata = _read_json_object(raw_metadata_path) if raw_metadata_path.exists() else {}
    dry_run = dry_run_vllm_planner_probe(
        translated,
        raw_safe_metadata=raw_metadata,
    ).to_dict()
    manifest: JsonObject = {
        "artifact": "vllm-planner-dry-run-probe",
        "timestamp": timestamp,
        "status": dry_run["status"],
        "source_translated_cache_config": str(translated_path),
        "source_raw_safe_metadata": str(raw_metadata_path) if raw_metadata_path.exists() else None,
        "safe_for_advisory_only": dry_run["safe_for_advisory_only"],
        "returned_to_vllm": False,
        "vllm_behavior_changed": False,
        "scheduler_mutation": False,
        "allocator_replacement": False,
        "worker_layout_mutation": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dry_run_result.json").write_text(
        json.dumps(dry_run, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(_summary(dry_run, manifest))
    (output_dir / "README.md").write_text(_readme(manifest))
    return {
        "manifest": manifest,
        "dry_run_result": dry_run,
    }


def _summary(dry_run: JsonObject, manifest: JsonObject) -> str:
    coverage = _as_json_object(dry_run["planner_input_coverage"])
    missing_mutation = "\n".join(
        f"- {item}" for item in _as_sequence(dry_run["missing_fields_that_prevent_mutation"])
    )
    return f"""# vLLM Planner Dry-Run Probe Summary

Status: `{manifest["status"]}`

This artifact is a planner-level dry run. It consumes the translated vanilla
vLLM cache-plan observation, computes a Cachepawl proposed planner view beside
the vanilla plan, and does not return that proposal to vLLM.

No vLLM source edits, monkeypatching, allocator replacement, scheduler
mutation, worker layout mutation, long-lived serving, Triton kernels, copy
kernels, LSDR, serving changes, or quality evaluation were performed.

## Key Metrics

- `vanilla_observed_reserved_bytes`: {dry_run["vanilla_observed_reserved_bytes"]}
- `vanilla_observed_useful_bytes`: {dry_run["vanilla_observed_useful_bytes"]}
- `cachepawl_proposed_reserved_bytes`: {dry_run["cachepawl_proposed_reserved_bytes"]}
- `estimated_savings_bytes`: {dry_run["estimated_savings_bytes"]}
- `overestimation_ratio`: {dry_run["overestimation_ratio"]}
- `wasted_fraction`: {dry_run["wasted_fraction"]}
- `safe_for_advisory_only`: {str(dry_run["safe_for_advisory_only"]).lower()}
- `returned_to_vllm`: {str(dry_run["returned_to_vllm"]).lower()}
- `vllm_behavior_changed`: {str(dry_run["vllm_behavior_changed"]).lower()}

## Planner Input Coverage

{_coverage_table(coverage)}

## Missing For Mutation

{missing_mutation}
"""


def _readme(manifest: JsonObject) -> str:
    return f"""# vLLM Planner Dry-Run Probe

Status: `{manifest["status"]}`

This artifact records a non-mutating Cachepawl proposed planner view computed
from translated vanilla vLLM runtime cache planning output.

Files:

- `manifest.json` — source paths and non-mutation flags.
- `dry_run_result.json` — structured dry-run metrics and proposed plan view.
- `summary.md` — concise human-readable summary.
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
