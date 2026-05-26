"""Command-line entrypoint for Cachepawl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cachepawl.integrations.vllm.diagnose import (
    VllmDiagnosticError,
    create_vllm_artifact_diagnostic,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "diagnose-vllm":
        try:
            create_vllm_artifact_diagnostic(
                translated_cache_config_path=args.translated_cache_config,
                raw_safe_metadata_path=args.raw_safe_metadata,
                output_dir=args.output_dir,
                timestamp=args.timestamp,
            )
        except VllmDiagnosticError as exc:
            print(f"cachepawl diagnose-vllm: error: {exc}", file=sys.stderr)
            return 2
        return 0

    parser.print_help(sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cachepawl", description="Cachepawl command line tools.")
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
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
