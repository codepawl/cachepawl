"""Smoke test for the kernels submodule placeholder."""

from __future__ import annotations

from cachepawl import kernels


def test_kernels_module_imports_cleanly() -> None:
    assert kernels.__all__ == []
