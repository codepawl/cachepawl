"""Tests for :mod:`cachepawl.kernels.allocate`.

The kernel body and :func:`launch_zero_page` wrapper landed in Week 1
of ``research/avmp/v2/TRITON_ROADMAP.md``. The GPU-marked tests verify
end-to-end correctness on a CUDA device; the boundary-case test
verifies the launcher's input validation independent of CUDA.
"""

from __future__ import annotations

import pytest
import torch

from cachepawl.kernels import allocate as allocate_module
from cachepawl.kernels.allocate import launch_zero_page, zero_page_kernel


def test_module_exports_kernel_and_launch_wrapper() -> None:
    """Both public symbols are present so allocator callers can wire."""

    assert hasattr(allocate_module, "zero_page_kernel")
    assert hasattr(allocate_module, "launch_zero_page")
    assert callable(zero_page_kernel)
    assert callable(launch_zero_page)


def test_launch_zero_page_validates_inputs() -> None:
    """Boundary checks fire on misuse before the kernel dispatch.

    Runs on CPU because the launcher rejects non-CUDA buffers before
    any kernel launch. The four scenarios cover device, dtype, range,
    and zero-size short-circuit.
    """

    cpu_buffer = torch.empty(4096, dtype=torch.uint8)
    with pytest.raises(ValueError, match="CUDA tensor"):
        launch_zero_page(cpu_buffer, offset=0, size_bytes=4096)

    if torch.cuda.is_available():
        wrong_dtype = torch.empty(4096, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match=r"torch\.uint8"):
            launch_zero_page(wrong_dtype, offset=0, size_bytes=4096)

        gpu_buffer = torch.empty(4096, dtype=torch.uint8, device="cuda")
        with pytest.raises(ValueError, match="non-negative"):
            launch_zero_page(gpu_buffer, offset=-1, size_bytes=4096)
        with pytest.raises(ValueError, match="exceeds buffer"):
            launch_zero_page(gpu_buffer, offset=2048, size_bytes=4096)

        # size_bytes == 0 is a no-op, must not raise
        launch_zero_page(gpu_buffer, offset=0, size_bytes=0)


@pytest.mark.gpu
def test_zero_page_kernel_full_buffer() -> None:
    """Zero-fill over the entire buffer leaves all bytes at 0.

    The simplest correctness case: a 4 KiB buffer pre-filled with 0xFF,
    zero-filled in one launch, all bytes must read back as 0 after a
    CUDA sync.
    """

    buf = torch.full((4096,), 0xFF, dtype=torch.uint8, device="cuda")
    launch_zero_page(buf, offset=0, size_bytes=4096)
    torch.cuda.synchronize()

    assert buf.eq(0).all().item()


@pytest.mark.gpu
def test_zero_page_kernel_offset() -> None:
    """A mid-buffer window zero-fills without touching neighbors.

    The contract that protects AVMP-allocated pages: a launch over
    ``[offset, offset + size)`` must not perturb bytes outside that
    range. AVMP page neighbors hold other live allocations, so any
    leak across the boundary corrupts other handles.
    """

    buf = torch.full((4096,), 0xFF, dtype=torch.uint8, device="cuda")
    launch_zero_page(buf, offset=1024, size_bytes=1024)
    torch.cuda.synchronize()

    assert buf[:1024].eq(0xFF).all().item(), "prefix corrupted"
    assert buf[1024:2048].eq(0).all().item(), "window not zeroed"
    assert buf[2048:].eq(0xFF).all().item(), "suffix corrupted"


@pytest.mark.gpu
def test_zero_page_kernel_non_aligned_size() -> None:
    """The tail-mask drops past-end threads in the last program.

    ``size_bytes`` need not be a multiple of ``BLOCK_SIZE``: AVMP page
    sizes derive from a model spec and are 128-byte aligned, but
    nothing guarantees 1024 alignment. The tail program writes only
    ``size_bytes - (n - 1) * BLOCK_SIZE`` bytes via the kernel mask.
    """

    buf = torch.full((4096,), 0xFF, dtype=torch.uint8, device="cuda")
    launch_zero_page(buf, offset=0, size_bytes=1500)
    torch.cuda.synchronize()

    assert buf[:1500].eq(0).all().item(), "head not zeroed"
    assert buf[1500:].eq(0xFF).all().item(), "past-end overwritten"


@pytest.mark.gpu
def test_zero_page_kernel_large() -> None:
    """Multi-program grid stress: a 16 MiB region requires many launches.

    16 MiB / 1024 B = 16,384 programs; this exceeds the RTX 3060's
    28 SM count by enough margin to verify the grid scheduling rather
    than just the single-program path.
    """

    buf = torch.full((16 * 1024 * 1024,), 0xFF, dtype=torch.uint8, device="cuda")
    launch_zero_page(buf, offset=0, size_bytes=16 * 1024 * 1024)
    torch.cuda.synchronize()

    assert buf.eq(0).all().item()
