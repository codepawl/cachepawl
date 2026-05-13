"""Verify the package exposes a non-empty version string."""

from __future__ import annotations

import cachepawl


def test_version_is_non_empty_string() -> None:
    assert isinstance(cachepawl.__version__, str)
    assert cachepawl.__version__
