#!/usr/bin/env python
"""Check Cachepawl package version consistency."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
UV_LOCK = ROOT / "uv.lock"
PACKAGE_NAME = "cachepawl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Optional git tag to compare, such as vX.Y.Z.")
    parser.add_argument(
        "--print-version",
        action="store_true",
        help="Print the canonical package version after checks pass.",
    )
    args = parser.parse_args(argv)

    version = read_pyproject_version()
    lock_version = read_uv_lock_editable_version()
    errors: list[str] = []

    if lock_version != version:
        errors.append(f"uv.lock has {lock_version!r}, but pyproject.toml has {version!r}")

    if args.tag is not None:
        expected_tag = f"v{version}"
        if args.tag != expected_tag:
            errors.append(f"tag {args.tag!r} does not match expected tag {expected_tag!r}")

    if errors:
        for error in errors:
            print(f"version consistency error: {error}", file=sys.stderr)
        return 1

    if args.print_version:
        print(version)
    return 0


def read_pyproject_version() -> str:
    text = PYPROJECT.read_text()
    project_match = re.search(r"(?ms)^\[project\]\n(?P<body>.*?)(?:^\[|\Z)", text)
    if project_match is None:
        raise RuntimeError("pyproject.toml does not contain a [project] table")
    return _read_quoted_version(project_match.group("body"), "pyproject.toml [project]")


def read_uv_lock_editable_version() -> str:
    text = UV_LOCK.read_text()
    for package_block in re.split(r"(?m)^\[\[package\]\]\n", text):
        if re.search(rf'^name = "{PACKAGE_NAME}"$', package_block, flags=re.MULTILINE):
            return _read_quoted_version(package_block, f"uv.lock {PACKAGE_NAME!r} package")
    raise RuntimeError(f"uv.lock does not contain an editable {PACKAGE_NAME!r} package")


def _read_quoted_version(text: str, source: str) -> str:
    version_match = re.search(r'^version = "(?P<version>[^"]+)"$', text, flags=re.MULTILINE)
    if version_match is None:
        raise RuntimeError(f"{source} does not contain a quoted version field")
    return version_match.group("version")


if __name__ == "__main__":
    raise SystemExit(main())
