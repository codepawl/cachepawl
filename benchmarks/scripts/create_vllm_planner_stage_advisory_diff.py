#!/usr/bin/env python
"""Create a non-mutating planner-stage advisory diff artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cachepawl.integrations.vllm import diff_vllm_planner_stage_advisory

DEFAULT_TRANSLATED_CONFIG = Path(
    "research/avmp/v2/results/vllm-planner-stage-observation/translated_planner_stage_config.json"
)
DEFAULT_OUTPUT_DIR = Path("research/avmp/v2/results/vllm-planner-stage-advisory-diff")
JsonObject = dict[str, object]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--translated-planner-stage-config",
        type=Path,
        default=DEFAULT_TRANSLATED_CONFIG,
    )
    parser.add_argument("--raw-safe-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=_timestamp())
    args = parser.parse_args(argv)

    create_planner_stage_advisory_diff(
        translated_planner_stage_config_path=args.translated_planner_stage_config,
        raw_safe_metadata_path=args.raw_safe_metadata,
        output_dir=args.output_dir,
        timestamp=args.timestamp,
    )
    return 0


def create_planner_stage_advisory_diff(
    *,
    translated_planner_stage_config_path: Path,
    raw_safe_metadata_path: Path | None,
    output_dir: Path,
    timestamp: str,
) -> JsonObject:
    translated = _read_json_object(translated_planner_stage_config_path)
    raw_metadata_path = _default_raw_metadata_path(
        translated_planner_stage_config_path=translated_planner_stage_config_path,
        raw_safe_metadata_path=raw_safe_metadata_path,
    )
    raw_metadata = _read_json_object(raw_metadata_path) if raw_metadata_path is not None else {}
    diff = diff_vllm_planner_stage_advisory(
        translated,
        raw_safe_metadata=raw_metadata,
    ).to_dict()
    group_level_diff = _as_sequence(diff["group_level_diff"])
    manifest: JsonObject = {
        "artifact": "vllm-planner-stage-advisory-diff",
        "timestamp": timestamp,
        "status": diff["status"],
        "source_translated_planner_stage_config": str(translated_planner_stage_config_path),
        "source_raw_safe_metadata": (
            str(raw_metadata_path) if raw_metadata_path is not None else None
        ),
        "outputs": {
            "readme": "README.md",
            "manifest": "manifest.json",
            "diff_report": "diff_report.json",
            "summary": "summary.md",
            "group_level_diff": "group_level_diff.json" if group_level_diff else None,
        },
        "non_mutating": True,
        "advisory_only": True,
        "returned_to_vllm": False,
        "vllm_behavior_changed": False,
        "scheduler_mutation": False,
        "allocator_replacement": False,
        "worker_layout_mutation": False,
        "vllm_required": False,
        "gpu_required": False,
        "nvml_required": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diff_report.json").write_text(_json_dumps(diff))
    (output_dir / "manifest.json").write_text(_json_dumps(manifest))
    (output_dir / "summary.md").write_text(_summary(diff=diff, manifest=manifest))
    (output_dir / "README.md").write_text(_readme(manifest=manifest))
    if group_level_diff:
        (output_dir / "group_level_diff.json").write_text(
            json.dumps(group_level_diff, indent=2, sort_keys=True) + "\n"
        )
    return {
        "manifest": manifest,
        "diff_report": diff,
        "group_level_diff": group_level_diff,
    }


def _default_raw_metadata_path(
    *,
    translated_planner_stage_config_path: Path,
    raw_safe_metadata_path: Path | None,
) -> Path | None:
    if raw_safe_metadata_path is not None:
        return raw_safe_metadata_path
    sibling = translated_planner_stage_config_path.parent / "raw_safe_metadata.json"
    return sibling if sibling.exists() else None


def _summary(*, diff: JsonObject, manifest: JsonObject) -> str:
    coverage = _as_json_object(diff["planner_input_coverage"])
    missing_mutation = "\n".join(
        f"- {item}" for item in _as_sequence(diff["missing_fields_that_prevent_mutation"])
    )
    return f"""# vLLM Planner-Stage Advisory Diff Summary

Status: `{manifest["status"]}`

This artifact is a planner-stage post-call advisory diff. It consumes the
translated vanilla T002 planner-stage output, computes a Cachepawl proposed
planner result beside it, and does not return that proposal to vLLM.

No vLLM source edits, monkeypatching, allocator replacement, scheduler
mutation, worker layout mutation, long-lived serving, Triton kernels, copy
kernels, LSDR, serving changes, or quality evaluation were performed.

## Key Metrics

- `vanilla_reserved_bytes`: {diff["vanilla_reserved_bytes"]}
- `vanilla_useful_bytes`: {diff["vanilla_useful_bytes"]}
- `cachepawl_proposed_reserved_bytes`: {diff["cachepawl_proposed_reserved_bytes"]}
- `estimated_savings_bytes`: {diff["estimated_savings_bytes"]}
- `overestimation_ratio`: {diff["overestimation_ratio"]}
- `wasted_fraction`: {diff["wasted_fraction"]}
- `cache_group_count`: {diff["cache_group_count"]}
- `cache_tensor_count`: {diff["cache_tensor_count"]}
- `layer_count`: {diff["layer_count"]}
- `num_blocks`: {diff["num_blocks"]}
- `non_mutating`: {str(diff["non_mutating"]).lower()}
- `returned_to_vllm`: {str(diff["returned_to_vllm"]).lower()}
- `vllm_behavior_changed`: {str(diff["vllm_behavior_changed"]).lower()}

## Planner Input Coverage

{_coverage_table(coverage)}

## Missing Fields That Still Prevent Mutation

{missing_mutation}
"""


def _readme(*, manifest: JsonObject) -> str:
    return f"""# vLLM Planner-Stage Advisory Diff

Status: `{manifest["status"]}`

This artifact records a non-mutating Cachepawl proposed planner result computed
beside the vanilla T002 planner-stage output.

Files:

- `manifest.json` — source paths, output names, and non-mutation flags.
- `diff_report.json` — structured planner-stage advisory diff.
- `summary.md` — concise human-readable summary.
- `group_level_diff.json` — optional per-group diff when derivable.

The Cachepawl proposal is not returned to vLLM and does not change vanilla vLLM
behavior.
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


def _json_dumps(value: JsonObject) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
