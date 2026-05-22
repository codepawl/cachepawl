"""Triton kernel for physical byte copy used during AVMP pool migration.

Reference design: ``docs/designs/0002-dynamic-pool-rebalancing.md`` section
4.5 (concurrency / write barrier semantics) and
``research/avmp/v2/TRITON_ROADMAP.md`` section 2.

Why this kernel exists. The Python AVMP prototype's :meth:`_apply_rebalance`
shrinks the donor pool by lowering its active-capacity counter and grows
the recipient pool by raising its counter. No tensor data moves because
the underlying ``BackingStore._buffer`` is single per pool and the
already-live handles keep their original offsets. A cuMemMap-backed
realization may instead need to physically relocate a live region (for
example when the donor pool's virtual-address range is unmapped and
returned to the driver). ``copy_region_kernel`` is the kernel that handles
that physical copy under the v2.1 design if and when that path is needed;
the Week 1 implementation may discover that the simpler counter-only
semantic carries over to hardware and the kernel stays unused. The slot
is scaffolded now to keep the roadmap honest about what could fail.

This module is scaffold-only. The body of :func:`copy_region_kernel` is
a no-op ``pass`` and :func:`launch_copy_region` raises
:class:`NotImplementedError`. The decision to use or skip this kernel is
recorded in ``research/avmp/v2/TRITON_ROADMAP.md`` section 1 once Week 2
lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import triton
import triton.language as tl

if TYPE_CHECKING:
    import torch


__all__ = ["copy_region_kernel", "launch_copy_region"]


# See note in ``allocate.py``: the @triton.jit decorator is flagged as
# untyped by mypy strict mode and silenced with a single ignore.
@triton.jit  # type: ignore[untyped-decorator]
def copy_region_kernel(  # type: ignore[no-untyped-def]
    src_ptr,
    dst_ptr,
    num_bytes,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Copy ``num_bytes`` bytes from ``src_ptr`` to ``dst_ptr``.

    Scaffold body. The Week 2 implementation (if the kernel is needed)
    will issue masked ``tl.load`` / ``tl.store`` pairs over a 1D grid.
    """

    # TODO(Week 2): implement the per-program coalesced copy.
    pass


def launch_copy_region(
    src: torch.Tensor,
    dst: torch.Tensor,
    num_bytes: int,
    block_size: int = 4096,
) -> None:
    """Launch :func:`copy_region_kernel` between two CUDA ``uint8`` buffers.

    ``src`` and ``dst`` must both be contiguous ``torch.uint8`` tensors on
    the same CUDA device. ``num_bytes`` must be ``<= min(src.numel(),
    dst.numel())``.

    Per RFC 0002 section 4.5, the caller is responsible for inserting any
    write barrier between the donor-pool shrink and the recipient-pool
    grow that brackets this copy.

    Raises :class:`NotImplementedError` until the Week 2 milestone lands
    (or until the decision to skip the kernel is recorded in the
    roadmap).
    """

    raise NotImplementedError(
        "launch_copy_region lands in Week 2 (or stays unused); "
        "see research/avmp/v2/TRITON_ROADMAP.md section 2"
    )
