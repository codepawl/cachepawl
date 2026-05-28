"""Verify the package exposes a non-empty version string."""

from __future__ import annotations

import subprocess
import sys

import cachepawl


def test_version_is_non_empty_string() -> None:
    assert isinstance(cachepawl.__version__, str)
    assert cachepawl.__version__


def test_public_version_matches_canonical_project_version() -> None:
    version = subprocess.check_output(
        [sys.executable, "scripts/check_version_consistency.py", "--print-version"],
        text=True,
    ).strip()
    assert cachepawl.__version__ == version


def test_version_consistency_script_passes() -> None:
    subprocess.run([sys.executable, "scripts/check_version_consistency.py"], check=True)
