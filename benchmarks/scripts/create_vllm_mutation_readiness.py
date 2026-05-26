#!/usr/bin/env python
"""Create a pre-mutation readiness artifact from vLLM planner artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cachepawl.integrations.vllm import check_vllm_mutation_readiness

DEFAULT_PLANNER_CONFIG = Path(
    "research/avmp/v2/results/vllm-planner-stage-observation/translated_planner_stage_config.json"
)
DEFAULT_DIFF_REPORT = Path(
    "research/avmp/v2/results/vllm-planner-stage-advisory-diff/diff_report.json"
)
DEFAULT_GROUP_LEVEL_DIFF = Path(
    "research/avmp/v2/results/vllm-planner-stage-advisory-diff/group_level_diff.json"
)
DEFAULT_OUTPUT_DIR = Path("research/avmp/v2/results/vllm-mutation-readiness")
JsonObject = dict[str, object]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--translated-planner-stage-config",
        type=Path,
        default=DEFAULT_PLANNER_CONFIG,
    )
    parser.add_argument("--diff-report", type=Path, default=DEFAULT_DIFF_REPORT)
    parser.add_argument("--group-level-diff", type=Path, default=DEFAULT_GROUP_LEVEL_DIFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=_timestamp())
    args = parser.parse_args(argv)

    create_mutation_readiness_artifact(
        translated_planner_stage_config_path=args.translated_planner_stage_config,
        diff_report_path=args.diff_report,
        group_level_diff_path=args.group_level_diff,
        output_dir=args.output_dir,
        timestamp=args.timestamp,
    )
    return 0


def create_mutation_readiness_artifact(
    *,
    translated_planner_stage_config_path: Path,
    diff_report_path: Path,
    group_level_diff_path: Path | None,
    output_dir: Path,
    timestamp: str,
) -> JsonObject:
    planner_config = _read_json_object(translated_planner_stage_config_path)
    diff_report = _read_json_object(diff_report_path)
    group_level_diff: object | None = (
        _read_json_array(group_level_diff_path)
        if group_level_diff_path is not None and group_level_diff_path.exists()
        else None
    )
    report = check_vllm_mutation_readiness(
        planner_config,
        diff_report,
        group_level_diff=group_level_diff,
    ).to_dict()
    manifest: JsonObject = {
        "artifact": "vllm-mutation-readiness",
        "timestamp": timestamp,
        "status": report["classification"],
        "source_translated_planner_stage_config": str(translated_planner_stage_config_path),
        "source_diff_report": str(diff_report_path),
        "source_group_level_diff": (
            str(group_level_diff_path)
            if group_level_diff_path is not None and group_level_diff_path.exists()
            else None
        ),
        "outputs": {
            "readme": "README.md",
            "manifest": "manifest.json",
            "readiness_report": "readiness_report.json",
            "summary": "summary.md",
        },
        "non_mutating": True,
        "returned_to_vllm": False,
        "vllm_behavior_changed": False,
        "vllm_required": False,
        "gpu_required": False,
        "nvml_required": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "readiness_report.json").write_text(_json_dumps(report))
    (output_dir / "manifest.json").write_text(_json_dumps(manifest))
    (output_dir / "summary.md").write_text(_summary(report=report, manifest=manifest))
    (output_dir / "README.md").write_text(_readme(manifest=manifest))
    return {
        "manifest": manifest,
        "readiness_report": report,
    }


def _summary(*, report: JsonObject, manifest: JsonObject) -> str:
    passed = "\n".join(f"- {item}" for item in _as_sequence(report["passed_invariants"]))
    failed = (
        "\n".join(f"- {item}" for item in _as_sequence(report["failed_invariants"])) or "- none"
    )
    blocked = (
        "\n".join(f"- {item}" for item in _as_sequence(report["blocked_invariants"])) or "- none"
    )
    missing = "\n".join(
        f"- {item}" for item in _as_sequence(report["mutation_required_missing_fields"])
    )
    return f"""# vLLM Mutation Readiness Summary

Status: `{manifest["status"]}`

This artifact is a pre-mutation safety gate. It validates existing serialized
planner-stage and advisory-diff artifacts without importing vLLM, using a GPU,
or returning Cachepawl plans to vLLM.

No vLLM source edits, monkeypatching, allocator replacement, scheduler
mutation, worker layout mutation, Triton kernels, copy kernels, LSDR, serving
changes, or quality evaluation were performed.

## Result

- `classification`: {report["classification"]}
- `ready_for_controlled_substitution`: {str(report["ready_for_controlled_substitution"]).lower()}
- `advisory_only`: {str(report["advisory_only"]).lower()}
- `non_mutating`: {str(report["non_mutating"]).lower()}
- `returned_to_vllm`: {str(report["returned_to_vllm"]).lower()}
- `vllm_behavior_changed`: {str(report["vllm_behavior_changed"]).lower()}

## Passed Invariants

{passed}

## Failed Invariants

{failed}

## Blocked Invariants

{blocked}

## Mutation-Required Missing Fields

{missing}
"""


def _readme(*, manifest: JsonObject) -> str:
    return f"""# vLLM Mutation Readiness

Status: `{manifest["status"]}`

This artifact checks whether the existing planner-stage and advisory-diff
artifacts are ready for a future controlled substitution experiment. It is
non-mutating and does not require vLLM, GPU, or NVML.

Files:

- `manifest.json` — source paths, output names, and non-mutation flags.
- `readiness_report.json` — structured compatibility and readiness checks.
- `summary.md` — concise human-readable summary.
"""


def _read_json_object(path: Path) -> JsonObject:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object from {path}, got {type(data).__name__}")
    return data


def _read_json_array(path: Path) -> tuple[object, ...]:
    data = json.loads(path.read_text())
    if isinstance(data, str):
        raise TypeError(f"expected JSON array from {path}, got string")
    try:
        return tuple(data)
    except TypeError as exc:
        raise TypeError(f"expected JSON array from {path}") from exc


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
