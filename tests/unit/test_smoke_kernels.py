"""Smoke test for the kernels submodule."""

from __future__ import annotations

from cachepawl import kernels


def test_kernels_module_imports_cleanly() -> None:
    assert set(kernels.__all__) == {
        "KernelLaunchConfig",
        "copy_region_kernel",
        "launch_copy_region",
        "launch_zero_page",
        "zero_page_kernel",
    }
