"""Command-line entrypoint for Cachepawl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cachepawl import __version__
from cachepawl.integrations.vllm.diagnose import (
    VllmDiagnosticError,
    create_vllm_artifact_diagnostic,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "diagnose-vllm":
        try:
            diagnostic = create_vllm_artifact_diagnostic(
                translated_cache_config_path=args.translated_cache_config,
                raw_safe_metadata_path=args.raw_safe_metadata,
                output_dir=args.output_dir,
                timestamp=args.timestamp,
            )
        except VllmDiagnosticError as exc:
            print(f"cachepawl diagnose-vllm: error: {exc}", file=sys.stderr)
            return 2
        if args.summary_only:
            if args.format == "json":
                sys.stdout.write(json.dumps(diagnostic.report, indent=2, sort_keys=True))
                sys.stdout.write("\n")
            else:
                sys.stdout.write(diagnostic.summary)
        threshold_failures = _diagnostic_threshold_failures(
            diagnostic.report,
            fail_on_waste_fraction=args.fail_on_waste_fraction,
            fail_on_overestimation_ratio=args.fail_on_overestimation_ratio,
        )
        if threshold_failures:
            for failure in threshold_failures:
                print(f"cachepawl diagnose-vllm: threshold failed: {failure}", file=sys.stderr)
            return 1
        return 0

    parser.print_help(sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cachepawl", description="Cachepawl command line tools.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    diagnose = subparsers.add_parser(
        "diagnose-vllm",
        help="Create a vLLM cache diagnostic from translated observation artifacts.",
    )
    diagnose.add_argument(
        "--translated-cache-config",
        type=Path,
        required=True,
        help="Path to translated_runtime_cache_config.json.",
    )
    diagnose.add_argument(
        "--raw-safe-metadata",
        type=Path,
        default=None,
        help="Optional path to raw_safe_metadata.json.",
    )
    diagnose.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where report.json, summary.md, and manifest.json are written.",
    )
    diagnose.add_argument(
        "--timestamp",
        default=None,
        help="Optional deterministic timestamp to record in manifest.json.",
    )
    diagnose.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the generated diagnostic summary to stdout after writing output files.",
    )
    diagnose.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="Stdout format for --summary-only. Output files are always JSON and Markdown.",
    )
    diagnose.add_argument(
        "--fail-on-waste-fraction",
        type=_nonnegative_float,
        default=None,
        metavar="FLOAT",
        help="Exit 1 when report.wasted_fraction is greater than this value.",
    )
    diagnose.add_argument(
        "--fail-on-overestimation-ratio",
        type=_nonnegative_float,
        default=None,
        metavar="FLOAT",
        help="Exit 1 when report.overestimation_ratio is greater than this value.",
    )
    return parser


def _diagnostic_threshold_failures(
    report: dict[str, object],
    *,
    fail_on_waste_fraction: float | None,
    fail_on_overestimation_ratio: float | None,
) -> list[str]:
    failures = []
    if fail_on_waste_fraction is not None and _metric_crosses_threshold(
        report["wasted_fraction"], fail_on_waste_fraction
    ):
        failures.append(f"wasted_fraction {report['wasted_fraction']} > {fail_on_waste_fraction}")
    if fail_on_overestimation_ratio is not None and _metric_crosses_threshold(
        report["overestimation_ratio"], fail_on_overestimation_ratio
    ):
        failures.append(
            "overestimation_ratio "
            f"{report['overestimation_ratio']} > {fail_on_overestimation_ratio}"
        )
    return failures


def _metric_crosses_threshold(value: object, threshold: float) -> bool:
    if value is None:
        return False
    if not isinstance(value, int | float):
        raise TypeError(f"diagnostic metric must be numeric or null, got {type(value).__name__}")
    return float(value) > threshold


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
