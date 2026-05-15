"""Asymmetric Virtual Memory Paging (AVMP) allocator.

See ``docs/designs/0001-asymmetric-virtual-memory-paging.md`` for the
full design rationale. This subpackage carries the v1 allocator
(:class:`AsymmetricVirtualPool`) plus its primitives (handle
dataclass, two physical backing stores, virtual page table). Dynamic
cross-pool rebalancing lands in a follow-up milestone.
"""

from cachepawl.allocator.avmp.pool import AsymmetricVirtualPool

__all__ = ["AsymmetricVirtualPool"]
