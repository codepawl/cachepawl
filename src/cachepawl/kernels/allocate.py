"""Triton kernel for zero-filling freshly allocated KV-page or SSM-block regions.

Reference design: ``docs/designs/0001-asymmetric-virtual-memory-paging.md``
section 5 (kernel-side resolution costs) and
``research/avmp/v2/TRITON_ROADMAP.md`` section 2.

Why this kernel exists. The Python AVMP prototype hands a caller a
freshly allocated handle whose backing offset inside
:class:`cachepawl.allocator.baselines.common.BackingStore` contains
uninitialized bytes (whatever the buffer held before the matching
:meth:`free_one` reclaimed the page). The simulation never reads that
memory so the staleness is invisible; a real LLM inference engine using
the allocated KV page does read it. ``zero_page_kernel`` keeps the
post-``allocate`` contract honest on hardware.

Week 1 implementation (TRITON_ROADMAP.md section 5):

- :func:`zero_page_kernel` is a 1-D scatter that writes
  ``size_bytes`` zero bytes starting at ``buffer_ptr + offset``.
- :func:`launch_zero_page` validates inputs at the system boundary
  and dispatches the kernel with ``BLOCK_SIZE = 1024``.

The launcher does not synchronize; callers (allocator, tests) sync
when they need to observe the writes from CPU code. Downstream GPU
kernels queued on the default stream see the zero-fill without an
explicit barrier.
"""

from __future__ import annotations

from typing import Final

import torch
import triton
import triton.language as tl

__all__ = ["launch_zero_page", "zero_page_kernel"]


_BLOCK_SIZE: Final[int] = 1024
"""Bytes per Triton program. RTX 3060 (CC 8.6) handles 1024 threads/block
comfortably; tunable per page-size via ``@triton.autotune`` once we have
the latency micro-bench data from the v2 paper."""


# Triton kernel parameters are JIT-time pointers and constexprs, not regular
# Python values; the @triton.jit decorator turns the function into an opaque
# callable. mypy strict mode flags the decorator as untyped and the inner
# function as missing annotations; both are silenced here. Each Triton
# program (one per ``triton.cdiv(size_bytes, BLOCK_SIZE)`` instance) writes
# ``BLOCK_SIZE`` contiguous bytes; the mask handles the tail.
@triton.jit  # type: ignore[untyped-decorator]
def zero_page_kernel(  # type: ignore[no-untyped-def]
    buffer_ptr,
    offset,
    size_bytes,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Zero-fill ``size_bytes`` bytes of ``buffer_ptr`` starting at ``offset``.

    Thread ``i`` of program ``pid`` writes byte ``offset + pid *
    BLOCK_SIZE + i``, so warps write contiguous bytes (fully coalesced).
    The mask drops threads past the page's last byte for the tail
    program when ``size_bytes`` is not a multiple of ``BLOCK_SIZE``.
    """

    pid = tl.program_id(axis=0)
    block_start = offset + pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < (offset + size_bytes)
    zeros = tl.zeros((BLOCK_SIZE,), dtype=tl.uint8)
    tl.store(buffer_ptr + offs, zeros, mask=mask)


def launch_zero_page(
    buffer: torch.Tensor,
    offset: int,
    size_bytes: int,
) -> None:
    """Launch :func:`zero_page_kernel` over a region of ``buffer``.

    ``buffer`` must be a contiguous ``torch.uint8`` tensor on a CUDA
    device (the live :class:`BackingStore._buffer`). ``offset`` and
    ``size_bytes`` must lie inside ``buffer.numel()``. ``size_bytes ==
    0`` returns without launching.

    No synchronization. Callers that need to observe the writes from
    CPU code must call ``torch.cuda.synchronize()`` (or the equivalent
    for non-default streams). Downstream kernels on the default stream
    see the zero-fill via implicit stream ordering.

    Validation happens here because this is the system boundary:
    direct test callers and any future code path that drives the kernel
    outside :class:`TritonAVMPAllocator` need their inputs checked.
    Allocator-side calls pass invariants guaranteed by
    :class:`AsymmetricVirtualPool`, so the checks are belt-and-braces
    for boundary callers.
    """

    if not buffer.is_cuda:
        raise ValueError(f"launch_zero_page requires a CUDA tensor, got device {buffer.device}")
    if buffer.dtype is not torch.uint8:
        raise ValueError(f"launch_zero_page requires dtype torch.uint8, got {buffer.dtype}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    if size_bytes < 0:
        raise ValueError(f"size_bytes must be non-negative, got {size_bytes}")
    if offset + size_bytes > buffer.numel():
        raise ValueError(
            f"region [{offset}, {offset + size_bytes}) exceeds buffer of {buffer.numel()} bytes"
        )
    if size_bytes == 0:
        return
    grid = (triton.cdiv(size_bytes, _BLOCK_SIZE),)
    zero_page_kernel[grid](buffer, offset, size_bytes, BLOCK_SIZE=_BLOCK_SIZE)
