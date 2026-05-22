"""Placeholder tests for :mod:`cachepawl.kernels.migrate`.

The kernel body and :func:`launch_copy_region` wrapper land in Week 2 of
``research/avmp/v2/TRITON_ROADMAP.md`` (or are explicitly removed if the
cuMemMap-backed migration path turns out not to need physical
relocation). Until then this module verifies the scaffold imports
cleanly and that the public wrapper refuses to run.
"""

from __future__ import annotations

import pytest
import torch

from cachepawl.kernels import migrate as migrate_module
from cachepawl.kernels.migrate import copy_region_kernel, launch_copy_region


def test_module_exports_kernel_and_launch_wrapper() -> None:
    """Both public symbols are present so Week 2 can wire callers."""

    assert hasattr(migrate_module, "copy_region_kernel")
    assert hasattr(migrate_module, "launch_copy_region")
    assert callable(copy_region_kernel)
    assert callable(launch_copy_region)


def test_launch_copy_region_raises_until_week2_lands() -> None:
    """Scaffold must raise, not silently return.

    A silent no-op would let a migration appear to succeed while leaving
    the recipient buffer holding the donor's pre-migration bytes. Fail
    loud per project rule 12.
    """

    src = torch.empty(8192, dtype=torch.uint8)
    dst = torch.empty(8192, dtype=torch.uint8)
    with pytest.raises(NotImplementedError, match="Week 2"):
        launch_copy_region(src, dst, num_bytes=4096)


@pytest.mark.gpu
@pytest.mark.skip(reason="kernel body lands in Week 2; see TRITON_ROADMAP.md section 2")
def test_copy_region_kernel_copies_full_region() -> None:
    """End-to-end byte-copy test placeholder.

    Week 2 will populate a CUDA ``torch.uint8`` source buffer with a
    deterministic pattern, run :func:`launch_copy_region` into a
    zero-initialized destination, then assert the destination matches
    the source over the targeted range and the surrounding bytes are
    unchanged.
    """
