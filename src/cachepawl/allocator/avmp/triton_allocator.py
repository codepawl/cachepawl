"""TritonAVMPAllocator: hardware realization of the AVMP design.

Reference design: ``docs/designs/0001-asymmetric-virtual-memory-paging.md``
(v1 AVMP), ``docs/designs/0002-dynamic-pool-rebalancing.md`` section 4.5
(concurrency / write barrier for cross-pool migration), and
``research/avmp/v2/TRITON_ROADMAP.md`` (3-week plan for the v2 paper at
ML for Systems @ NeurIPS 2026).

This subclass adds the physical GPU operations that
:class:`AsymmetricVirtualPool` lacks. The Python prototype carves views
into a pre-allocated ``torch.uint8`` buffer and never reads the bytes,
which is fine for simulation but not for a real LLM inference engine
that would observe stale data in a freshly allocated KV page. The
Triton variant zero-fills each allocated region via
:func:`cachepawl.kernels.launch_zero_page` (Week 1) and, if RFC 0002
section 4.5 ends up requiring physical relocation during migration,
copies live data via :func:`cachepawl.kernels.launch_copy_region`
(Week 2).

The state machine (:class:`PoolPressureState`), capacity-error retry
ladder (:meth:`AsymmetricVirtualPool._try_bulk_allocate_with_eviction`),
observability surface (:meth:`AsymmetricVirtualPool.get_allocator_stats`)
and all bookkeeping primitives (:class:`VirtualPageTable`,
:class:`KVPagesStore`, :class:`SSMBlocksStore`) are inherited unchanged
from the base class. Only the methods that actually launch kernels are
overridden, which keeps the diff focused on the new behavior.

Scaffold-only in this PR: every override calls ``super()`` and then
raises :class:`NotImplementedError`. The kernel-launch bodies land in
Week 1 / Week 2 of the roadmap.
"""

from __future__ import annotations

from cachepawl.allocator.avmp.handle import HandleKind
from cachepawl.allocator.avmp.pool import AsymmetricVirtualPool
from cachepawl.allocator.avmp.state import RebalanceDirection, RebalanceOutcome


class TritonAVMPAllocator(AsymmetricVirtualPool):
    """AVMP allocator backed by Triton kernels for the data-touching paths.

    Inheritance choice (recorded in ``research/avmp/v2/TRITON_ROADMAP.md``
    section 1): subclassing :class:`AsymmetricVirtualPool` rather than
    re-implementing the state machine keeps the donor-shrink /
    recipient-grow orchestration, the auto-trigger throttle, and the
    full :meth:`get_allocator_stats` surface aligned with the Python
    prototype by construction. The byte-for-byte parity smoke in the
    Week 2 milestone depends on this alignment.

    The constructor signature is identical to
    :class:`AsymmetricVirtualPool`; the device argument must be a CUDA
    device because the overridden methods launch Triton kernels. A CPU
    device will cause the Week 1 implementation to raise; the scaffold
    raises for any input.
    """

    def _allocate_into(self, *, kind: HandleKind, num_blocks: int) -> list[int]:
        """Allocate ``num_blocks`` pages or blocks and zero-fill each.

        Override seam (Week 1): after the base class returns the list of
        physical offsets via :meth:`_try_bulk_allocate_with_eviction`,
        the Triton variant launches :func:`launch_zero_page` over each
        offset's region of the matching :class:`BackingStore._buffer`
        before minting handles. Mint order is preserved so handle ids
        remain comparable to the Python baseline.
        """

        # TODO(Week 1): call super()._try_bulk_allocate_with_eviction
        # directly (skipping the base mint loop), launch zero_page_kernel
        # over each offset, then mint handles in the same order so handle
        # ids match the Python baseline byte-for-byte.
        raise NotImplementedError(
            "TritonAVMPAllocator._allocate_into lands in Week 1; see "
            "research/avmp/v2/TRITON_ROADMAP.md section 5"
        )

    def _apply_rebalance(
        self,
        direction: RebalanceDirection,
        batch_blocks: int,
    ) -> RebalanceOutcome:
        """Shrink the donor pool and grow the recipient pool.

        Override seam (Week 2, only if needed): the Python base class
        adjusts active-capacity counters without moving tensor data. The
        cuMemMap-backed realization may need to physically relocate live
        regions before the donor's address range is unmapped; if so,
        this override launches :func:`launch_copy_region` for each live
        region between the donor shrink and the recipient grow, per RFC
        0002 section 4.5 write-barrier semantics.

        If the counter-only semantic carries over to hardware, this
        override is removed and the base class implementation is used
        unchanged.
        """

        # TODO(Week 2): decide whether physical relocation is needed
        # under the cuMemMap design. If yes, launch copy_region_kernel
        # for each live page between donor shrink and recipient grow;
        # if no, delete this override.
        raise NotImplementedError(
            "TritonAVMPAllocator._apply_rebalance lands in Week 2 (or stays "
            "inherited from AsymmetricVirtualPool); see "
            "research/avmp/v2/TRITON_ROADMAP.md section 2"
        )
