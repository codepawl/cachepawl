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
the allocated KV page does read it. ``zero_page_kernel`` is the kernel
that lands on the v1 hardware realization to keep the post-``allocate``
contract honest.

This module is scaffold-only. The body of :func:`zero_page_kernel` is a
no-op ``pass`` and the public callable :func:`launch_zero_page` raises
:class:`NotImplementedError`. The kernel body lands in Week 1 of the
roadmap (see ``research/avmp/v2/TRITON_ROADMAP.md`` section 5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import triton
import triton.language as tl

if TYPE_CHECKING:
    import torch


__all__ = ["launch_zero_page", "zero_page_kernel"]


# Triton kernel parameters are JIT-time pointers and constexprs, not regular
# Python values; the @triton.jit decorator turns the function into an opaque
# callable. mypy strict mode flags the decorator as untyped, so the def is
# silenced with a single ignore on the decorator line.
@triton.jit  # type: ignore[untyped-decorator]
def zero_page_kernel(  # type: ignore[no-untyped-def]
    buffer_ptr,
    offset,
    size_bytes,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Zero-fill ``size_bytes`` bytes of ``buffer_ptr`` starting at ``offset``.

    Scaffold body. The Week 1 implementation will compute the per-program
    byte range, mask the tail block, and issue a coalesced ``tl.store``
    of ``tl.zeros((BLOCK_SIZE,), dtype=tl.uint8)``.
    """

    # TODO(Week 1): implement the per-program coalesced zero-fill.
    pass


def launch_zero_page(
    buffer: torch.Tensor,
    offset: int,
    size_bytes: int,
    block_size: int = 4096,
) -> None:
    """Launch :func:`zero_page_kernel` over a region of ``buffer``.

    ``buffer`` must be a contiguous ``torch.uint8`` tensor on a CUDA
    device (the live :class:`BackingStore._buffer`). ``offset`` and
    ``size_bytes`` must lie inside ``buffer.numel()``.

    ``block_size`` is the per-program byte width and is passed as the
    ``BLOCK_SIZE`` constexpr to the kernel. The default of 4096 matches
    the planned coalesced-store width on RTX 3060 (CC 8.6) per
    ``research/avmp/v2/TRITON_ROADMAP.md`` section 2.

    Raises :class:`NotImplementedError` until the Week 1 milestone lands.
    """

    raise NotImplementedError(
        "launch_zero_page lands in Week 1; see research/avmp/v2/TRITON_ROADMAP.md section 5"
    )
