#!/usr/bin/env python
"""Run or summarize a bounded vLLM Path C advisory-only matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CONFIG = Path("research/avmp/v2/evaluation/config_matrix.json")
DEFAULT_OUTPUT_DIR = Path("research/avmp/v2/evaluation")
DEFAULT_RESULTS_DIR = Path("research/avmp/v2/results/vllm-path-c-matrix")
DEFAULT_EXISTING_BASELINE_DIFF = Path(
    "research/avmp/v2/results/vllm-planner-stage-advisory-diff/diff_report.json"
)
PLANNER_STAGE_SCRIPT = Path("benchmarks/scripts/capture_vllm_planner_stage_observation.py")
ADVISORY_DIFF_SCRIPT = Path("benchmarks/scripts/create_vllm_planner_stage_advisory_diff.py")
JsonObject = dict[str, object]

MATRIX_COLUMNS = (
    "model",
    "max_model_len",
    "gpu_memory_utilization",
    "max_num_seqs",
    "vanilla_reserved_bytes",
    "vanilla_useful_bytes",
    "cachepawl_proposed_reserved_bytes",
    "estimated_savings_bytes",
    "overestimation_ratio",
    "wasted_fraction",
    "num_blocks",
    "cache_group_count",
    "cache_tensor_count",
    "layer_count",
    "status",
    "blocker",
)


@dataclass(frozen=True)
class MatrixPoint:
    model: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int


@dataclass(frozen=True)
class MatrixConfig:
    artifact: str
    model: str
    vllm_version: str
    pinned_venv_path: str
    results_dir: Path
    max_model_lens: tuple[int, ...]
    gpu_memory_utilizations: tuple[float, ...]
    max_num_seqs_values: tuple[int, ...]
    timeout_seconds: int
    trust_remote_code: bool


@dataclass(frozen=True)
class MatrixRow:
    point: MatrixPoint
    vanilla_reserved_bytes: int | None
    vanilla_useful_bytes: int | None
    cachepawl_proposed_reserved_bytes: int | None
    estimated_savings_bytes: int | None
    overestimation_ratio: float | None
    wasted_fraction: float | None
    num_blocks: int | None
    cache_group_count: int | None
    cache_tensor_count: int | None
    layer_count: int | None
    status: str
    blocker: str

    def to_csv_dict(self) -> dict[str, str]:
        values: dict[str, object | None] = {
            "model": self.point.model,
            "max_model_len": self.point.max_model_len,
            "gpu_memory_utilization": self.point.gpu_memory_utilization,
            "max_num_seqs": self.point.max_num_seqs,
            "vanilla_reserved_bytes": self.vanilla_reserved_bytes,
            "vanilla_useful_bytes": self.vanilla_useful_bytes,
            "cachepawl_proposed_reserved_bytes": self.cachepawl_proposed_reserved_bytes,
            "estimated_savings_bytes": self.estimated_savings_bytes,
            "overestimation_ratio": self.overestimation_ratio,
            "wasted_fraction": self.wasted_fraction,
            "num_blocks": self.num_blocks,
            "cache_group_count": self.cache_group_count,
            "cache_tensor_count": self.cache_tensor_count,
            "layer_count": self.layer_count,
            "status": self.status,
            "blocker": self.blocker,
        }
        return {column: _stringify_cell(values[column]) for column in MATRIX_COLUMNS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--existing-baseline-diff-report",
        type=Path,
        default=DEFAULT_EXISTING_BASELINE_DIFF,
        help="Existing single-config diff report used only when its matrix point matches.",
    )
    parser.add_argument("--run-observations", action="store_true")
    parser.add_argument("--timestamp", default=_timestamp())
    args = parser.parse_args(argv)

    config = load_matrix_config(args.config)
    results_dir = args.results_dir if args.results_dir is not None else config.results_dir
    if args.run_observations:
        run_matrix_observations(config=config, results_dir=results_dir, timestamp=args.timestamp)
    rows = matrix_rows_from_config(
        config=config,
        results_dir=results_dir,
        existing_baseline_diff_report=args.existing_baseline_diff_report,
    )
    write_matrix_artifacts(
        rows=rows,
        config=config,
        output_dir=args.output_dir,
        results_dir=results_dir,
    )
    return 0


def load_matrix_config(path: Path) -> MatrixConfig:
    data = _read_json_object(path)
    matrix = _as_json_object(data["matrix"])
    return MatrixConfig(
        artifact=_as_str(data["artifact"]),
        model=_as_str(data["model"]),
        vllm_version=_as_str(data["vllm_version"]),
        pinned_venv_path=_as_str(data["pinned_venv_path"]),
        results_dir=Path(_as_str(data.get("results_dir", str(DEFAULT_RESULTS_DIR)))),
        max_model_lens=tuple(_as_int(item) for item in _as_sequence(matrix["max_model_len"])),
        gpu_memory_utilizations=tuple(
            _as_float(item) for item in _as_sequence(matrix["gpu_memory_utilization"])
        ),
        max_num_seqs_values=tuple(_as_int(item) for item in _as_sequence(matrix["max_num_seqs"])),
        timeout_seconds=_as_int(data.get("timeout_seconds", 1200)),
        trust_remote_code=_as_bool(data.get("trust_remote_code", False)),
    )


def matrix_points(config: MatrixConfig) -> tuple[MatrixPoint, ...]:
    return tuple(
        MatrixPoint(
            model=config.model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
        )
        for max_model_len in config.max_model_lens
        for gpu_memory_utilization in config.gpu_memory_utilizations
        for max_num_seqs in config.max_num_seqs_values
    )


def matrix_rows_from_config(
    *,
    config: MatrixConfig,
    results_dir: Path,
    existing_baseline_diff_report: Path | None,
) -> list[MatrixRow]:
    baseline = (
        _read_json_object(existing_baseline_diff_report)
        if _exists(existing_baseline_diff_report)
        else None
    )
    rows: list[MatrixRow] = []
    for point in matrix_points(config):
        cell_dir = _cell_dir(results_dir, point)
        diff_report_path = cell_dir / "planner-stage-advisory-diff" / "diff_report.json"
        blocker_path = cell_dir / "planner-stage-observation" / "blocker.json"
        if diff_report_path.exists():
            rows.append(
                _row_from_diff_report(point, _read_json_object(diff_report_path), "completed")
            )
        elif _baseline_matches(point, baseline):
            rows.append(
                _row_from_diff_report(
                    point,
                    _as_json_object(baseline),
                    "completed_existing_baseline",
                )
            )
        elif blocker_path.exists():
            blocker = _read_json_object(blocker_path)
            rows.append(_blocked_row(point, "blocked", _blocker_reason(blocker)))
        else:
            rows.append(_blocked_row(point, "pending_not_run", "matrix point not run"))
    return rows


def run_matrix_observations(
    *,
    config: MatrixConfig,
    results_dir: Path,
    timestamp: str,
) -> None:
    for point in matrix_points(config):
        point_dir = _cell_dir(results_dir, point)
        observation_dir = point_dir / "planner-stage-observation"
        advisory_dir = point_dir / "planner-stage-advisory-diff"
        observation_dir.mkdir(parents=True, exist_ok=True)
        _write_cell_manifest(point_dir=point_dir, point=point, status="running")
        _run_command(
            [
                sys.executable,
                str(PLANNER_STAGE_SCRIPT),
                "--output-dir",
                str(observation_dir),
                "--timestamp",
                timestamp,
                "--model",
                point.model,
                "--max-model-len",
                str(point.max_model_len),
                "--gpu-memory-utilization",
                str(point.gpu_memory_utilization),
                "--max-num-seqs",
                str(point.max_num_seqs),
                "--timeout-seconds",
                str(config.timeout_seconds),
            ]
            + (["--trust-remote-code"] if config.trust_remote_code else [])
        )
        translated_path = observation_dir / "translated_planner_stage_config.json"
        if translated_path.exists():
            _run_command(
                [
                    sys.executable,
                    str(ADVISORY_DIFF_SCRIPT),
                    "--translated-planner-stage-config",
                    str(translated_path),
                    "--raw-safe-metadata",
                    str(observation_dir / "raw_safe_metadata.json"),
                    "--output-dir",
                    str(advisory_dir),
                    "--timestamp",
                    timestamp,
                ]
            )
            _write_cell_manifest(point_dir=point_dir, point=point, status="completed")
        else:
            _write_cell_manifest(point_dir=point_dir, point=point, status="blocked")


def write_matrix_artifacts(
    *,
    rows: list[MatrixRow],
    config: MatrixConfig,
    output_dir: Path,
    results_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        point_dir = _cell_dir(results_dir, row.point)
        point_dir.mkdir(parents=True, exist_ok=True)
        _write_cell_manifest(point_dir=point_dir, point=row.point, status=row.status)
        (point_dir / "README.md").write_text(_cell_readme(row))
    _write_csv(output_dir / "matrix_table.csv", rows)
    (output_dir / "matrix_table.md").write_text(_markdown_table(rows))


def _write_csv(path: Path, rows: list[MatrixRow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def _markdown_table(rows: list[MatrixRow]) -> str:
    lines = [
        "# vLLM Path C Advisory Matrix",
        "",
        "This table is advisory/diagnostic evidence only. It does not report runtime",
        "mutation, throughput, serving, or VRAM improvement measurements.",
        "",
        "| " + " | ".join(MATRIX_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in MATRIX_COLUMNS) + " |",
    ]
    for row in rows:
        cells = [row.to_csv_dict()[column] for column in MATRIX_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _cell_readme(row: MatrixRow) -> str:
    return f"""# vLLM Path C Matrix Cell

- Model: `{row.point.model}`
- `max_model_len`: `{row.point.max_model_len}`
- `gpu_memory_utilization`: `{row.point.gpu_memory_utilization}`
- `max_num_seqs`: `{row.point.max_num_seqs}`
- Status: `{row.status}`
- Blocker: `{row.blocker or "none"}`

This cell is part of the advisory-only vLLM Path C evaluation matrix. It does
not modify vLLM, replace allocators, monkeypatch scheduler behavior, or return a
Cachepawl plan to vLLM.
"""


def _write_cell_manifest(*, point_dir: Path, point: MatrixPoint, status: str) -> None:
    manifest: JsonObject = {
        "artifact": "vllm-path-c-advisory-matrix-cell",
        "model": point.model,
        "max_model_len": point.max_model_len,
        "gpu_memory_utilization": point.gpu_memory_utilization,
        "max_num_seqs": point.max_num_seqs,
        "status": status,
        "advisory_only": True,
        "non_mutating": True,
        "returned_to_vllm": False,
        "vllm_behavior_changed": False,
    }
    (point_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _row_from_diff_report(point: MatrixPoint, diff: JsonObject, status: str) -> MatrixRow:
    return MatrixRow(
        point=point,
        vanilla_reserved_bytes=_optional_int(diff.get("vanilla_reserved_bytes")),
        vanilla_useful_bytes=_optional_int(diff.get("vanilla_useful_bytes")),
        cachepawl_proposed_reserved_bytes=_optional_int(
            diff.get("cachepawl_proposed_reserved_bytes")
        ),
        estimated_savings_bytes=_optional_int(diff.get("estimated_savings_bytes")),
        overestimation_ratio=_optional_float(diff.get("overestimation_ratio")),
        wasted_fraction=_optional_float(diff.get("wasted_fraction")),
        num_blocks=_optional_int(diff.get("num_blocks")),
        cache_group_count=_optional_int(diff.get("cache_group_count")),
        cache_tensor_count=_optional_int(diff.get("cache_tensor_count")),
        layer_count=_optional_int(diff.get("layer_count")),
        status=status,
        blocker="",
    )


def _blocked_row(point: MatrixPoint, status: str, blocker: str) -> MatrixRow:
    return MatrixRow(
        point=point,
        vanilla_reserved_bytes=None,
        vanilla_useful_bytes=None,
        cachepawl_proposed_reserved_bytes=None,
        estimated_savings_bytes=None,
        overestimation_ratio=None,
        wasted_fraction=None,
        num_blocks=None,
        cache_group_count=None,
        cache_tensor_count=None,
        layer_count=None,
        status=status,
        blocker=blocker,
    )


def _baseline_matches(point: MatrixPoint, baseline: JsonObject | None) -> bool:
    if baseline is None:
        return False
    return (
        point.max_model_len == 4096
        and point.gpu_memory_utilization == 0.7
        and point.max_num_seqs == 1
    )


def _blocker_reason(blocker: JsonObject) -> str:
    reason = blocker.get("reason", blocker.get("blocker", "observation blocked"))
    return str(reason)


def _cell_dir(results_dir: Path, point: MatrixPoint) -> Path:
    gpu = str(point.gpu_memory_utilization).replace(".", "p")
    model = point.model.replace("/", "_").replace(":", "_")
    return results_dir / f"{model}__len{point.max_model_len}__gpu{gpu}__seqs{point.max_num_seqs}"


def _run_command(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        check=False,
        env={**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {_format_command(command)}"
        )


def _format_command(command: list[str]) -> str:
    return " ".join(command)


def _read_json_object(path: Path | None) -> JsonObject:
    if path is None:
        raise TypeError("expected path")
    return _as_json_object(json.loads(path.read_text()))


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


def _as_str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected string, got {type(value).__name__}")
    return value


def _as_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"expected bool, got {type(value).__name__}")
    return value


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int, got {type(value).__name__}")
    return value


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected float, got {type(value).__name__}")
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _as_int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _as_float(value)


def _stringify_cell(value: object | None) -> str:
    return "" if value is None else str(value)


def _exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
