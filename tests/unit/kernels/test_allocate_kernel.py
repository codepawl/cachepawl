"""Placeholder tests for :mod:`cachepawl.kernels.allocate`.

The kernel body and :func:`launch_zero_page` wrapper land in Week 1 of
``research/avmp/v2/TRITON_ROADMAP.md``. Until then this module only
verifies that the scaffold imports cleanly and that the public wrapper
still refuses to run (so a partial implementation cannot silently
return zero-filled garbage on review).
"""

from __future__ import annotations

import pytest
import torch

from cachepawl.kernels import allocate as allocate_module
from cachepawl.kernels.allocate import launch_zero_page, zero_page_kernel


def test_module_exports_kernel_and_launch_wrapper() -> None:
    """Both public symbols are present so Week 1 can wire callers."""

    assert hasattr(allocate_module, "zero_page_kernel")
    assert hasattr(allocate_module, "launch_zero_page")
    assert callable(zero_page_kernel)
    assert callable(launch_zero_page)


def test_launch_zero_page_raises_until_week1_lands() -> None:
    """Scaffold must raise, not silently return.

    A silent no-op would let a caller think the allocated region was
    zero-filled when it still holds whatever bytes the prior free left
    behind. Fail loud per project rule 12.
    """

    buffer = torch.empty(8192, dtype=torch.uint8)
    with pytest.raises(NotImplementedError, match="Week 1"):
        launch_zero_page(buffer, offset=0, size_bytes=4096)


@pytest.mark.gpu
@pytest.mark.skip(reason="kernel body lands in Week 1; see TRITON_ROADMAP.md section 5")
def test_zero_page_kernel_zeros_full_region() -> None:
    """End-to-end zero-fill test placeholder.

    Week 1 will fill a CUDA ``torch.uint8`` buffer with 0xFF, run
    :func:`launch_zero_page` over a sub-range, then assert the
    targeted bytes are 0 and the surrounding bytes are unchanged.
    """
