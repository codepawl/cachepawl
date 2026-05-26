"""Artifact-input vLLM cache diagnostics.

This module is import-safe without vLLM. It consumes serialized Cachepawl
runtime-observation artifacts and emits advisory-only diagnostic files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import TypeAlias

from cachepawl.integrations.vllm.advisory import advise_vllm_runtime_cache_plan
from cachepawl.integrations.vllm.dry_run import dry_run_vllm_planner_probe

JsonObject: TypeAlias = dict[str, object]


class VllmDiagnosticError(ValueError):
    """Raised for user-facing diagnostic CLI errors."""


@dataclass(frozen=True, slots=True)
class VllmArtifactDiagnostic:
    """Files and payloads emitted by artifact-input vLLM diagnostics."""

    report: JsonObject
    summary: str
    manifest: JsonObject
    output_dir: Path


def create_vllm_artifact_diagnostic(
    *,
    translated_cache_config_path: Path,
    raw_safe_metadata_path: Path | None,
    output_dir: Path,
    timestamp: str | None = None,
) -> VllmArtifactDiagnostic:
    """Create deterministic diagnostic output from translated runtime artifacts."""

    translated = _read_required_json_object(
        translated_cache_config_path,
        missing_message="missing translated config file",
    )
    raw_metadata = (
        _read_required_json_object(
            raw_safe_metadata_path, missing_message="missing raw metadata file"
        )
        if raw_safe_metadata_path is not None
        else {}
    )

    try:
        advisory = advise_vllm_runtime_cache_plan(
            translated,
            raw_safe_metadata=raw_metadata,
        ).to_dict()
        dry_run = dry_run_vllm_planner_probe(
            translated,
            raw_safe_metadata=raw_metadata,
        ).to_dict()
    except (TypeError, ValueError) as exc:
        raise VllmDiagnosticError(f"unsupported translated config schema: {exc}") from exc

    report = _report(
        advisory=advisory,
        dry_run=dry_run,
    )
    manifest = _manifest(
        report=report,
        translated_cache_config_path=translated_cache_config_path,
        raw_safe_metadata_path=raw_safe_metadata_path,
        timestamp=timestamp,
    )
    summary = _summary(report=report, manifest=manifest)
    _write_outputs(output_dir=output_dir, report=report, summary=summary, manifest=manifest)
    return VllmArtifactDiagnostic(
        report=report,
        summary=summary,
        manifest=manifest,
        output_dir=output_dir,
    )


def _report(*, advisory: JsonObject, dry_run: JsonObject) -> JsonObject:
    return {
        "artifact": "vllm-runtime-cache-diagnostic-cli",
        "mode": "artifact_input",
        "classification": advisory["primary_classification"],
        "classifications": advisory["classifications"],
        "advisory_only": True,
        "vllm_behavior_changed": False,
        "runtime_mutation": False,
        "allocator_replacement": False,
        "runtime_savings_require_future_mutation_hook": True,
        "num_blocks": advisory["num_blocks"],
        "cache_group_count": advisory["cache_group_count"],
        "cache_tensor_count": advisory["cache_tensor_count"],
        "layer_count": advisory["layer_count"],
        "available_kv_cache_gpu_memory_bytes": advisory["available_kv_cache_gpu_memory_bytes"],
        "observed_reserved_bytes": advisory["observed_reserved_bytes"],
        "observed_useful_bytes": advisory["observed_useful_bytes"],
        "cachepawl_recommended_bytes": advisory["cachepawl_recommended_bytes"],
        "advisory_savings_bytes": advisory["advisory_savings_bytes"],
        "overestimation_ratio": advisory["overestimation_ratio"],
        "wasted_fraction": advisory["wasted_fraction"],
        "planner_input_coverage": advisory["planner_input_coverage"],
        "missing_mutation_fields": advisory["missing_fields_for_mutation"],
        "group_advisories": advisory["group_advisories"],
        "dry_run": dry_run,
    }


def _manifest(
    *,
    report: JsonObject,
    translated_cache_config_path: Path,
    raw_safe_metadata_path: Path | None,
    timestamp: str | None,
) -> JsonObject:
    return {
        "artifact": "vllm-runtime-cache-diagnostic-cli",
        "mode": "artifact_input",
        "timestamp": timestamp,
        "status": report["classification"],
        "source_translated_cache_config": str(translated_cache_config_path),
        "source_raw_safe_metadata": (
            str(raw_safe_metadata_path) if raw_safe_metadata_path is not None else None
        ),
        "outputs": {
            "report": "report.json",
            "summary": "summary.md",
            "manifest": "manifest.json",
        },
        "advisory_only": True,
        "vllm_required": False,
        "cuda_required": False,
        "gpu_required": False,
        "nvml_required": False,
        "vllm_behavior_changed": False,
        "runtime_mutation": False,
        "allocator_replacement": False,
        "returned_to_vllm": False,
        "model_loaded": False,
    }


def _summary(*, report: JsonObject, manifest: JsonObject) -> str:
    classifications = ", ".join(str(item) for item in _as_sequence(report["classifications"]))
    missing_mutation = "\n".join(
        f"- {item}" for item in _as_sequence(report["missing_mutation_fields"])
    )
    return f"""# vLLM Runtime Cache Diagnostic

Status: `{manifest["status"]}`

This diagnostic is advisory-only. It reads translated Cachepawl runtime
observation artifacts and does not rerun vLLM, load a model, mutate runtime
state, replace allocators, or change vLLM behavior.

Runtime savings require a future mutation hook. This command only reports the
observed vLLM cache reservation and the Cachepawl advisory planner view.

## Classification

{classifications}

## Key Metrics

- `observed_reserved_bytes`: {report["observed_reserved_bytes"]}
- `observed_useful_bytes`: {report["observed_useful_bytes"]}
- `cachepawl_recommended_bytes`: {report["cachepawl_recommended_bytes"]}
- `advisory_savings_bytes`: {report["advisory_savings_bytes"]}
- `overestimation_ratio`: {report["overestimation_ratio"]}
- `wasted_fraction`: {report["wasted_fraction"]}
- `cache_group_count`: {report["cache_group_count"]}
- `cache_tensor_count`: {report["cache_tensor_count"]}
- `layer_count`: {report["layer_count"]}
- `num_blocks`: {report["num_blocks"]}
- `classification`: {report["classification"]}

## Runtime Safety

- `advisory_only`: true
- `vllm_behavior_changed`: false
- `runtime_mutation`: false
- `allocator_replacement`: false
- `runtime_savings_require_future_mutation_hook`: true
- `vllm_required`: false
- `gpu_required`: false
- `nvml_required`: false

## Missing Mutation Fields

{missing_mutation}
"""


def _write_outputs(
    *,
    output_dir: Path,
    report: JsonObject,
    summary: str,
    manifest: JsonObject,
) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(_json_dumps(report))
        (output_dir / "summary.md").write_text(summary)
        (output_dir / "manifest.json").write_text(_json_dumps(manifest))
    except OSError as exc:
        raise VllmDiagnosticError(f"output directory write failure: {output_dir}: {exc}") from exc


def _read_required_json_object(path: Path, *, missing_message: str) -> JsonObject:
    if not path.exists():
        raise VllmDiagnosticError(f"{missing_message}: {path}")
    try:
        data = json.loads(path.read_text())
    except JSONDecodeError as exc:
        raise VllmDiagnosticError(
            f"invalid JSON in {path}: line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise VllmDiagnosticError(f"failed to read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise VllmDiagnosticError(
            f"unsupported translated config schema: {path} must contain a JSON object"
        )
    return data


def _json_dumps(value: JsonObject) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _as_sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, str):
        raise TypeError("expected sequence, got string")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("expected sequence") from exc
