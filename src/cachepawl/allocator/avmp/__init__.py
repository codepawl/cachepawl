"""Asymmetric Virtual Memory Paging (AVMP) allocator.

See ``docs/designs/0001-asymmetric-virtual-memory-paging.md`` for the
v1 design rationale and
``docs/designs/0002-dynamic-pool-rebalancing.md`` for the v2 dynamic
rebalancing design. This subpackage carries the v1 allocator
(:class:`AsymmetricVirtualPool`) plus its primitives (handle dataclass,
two physical backing stores, virtual page table) and the v2 state-machine
scaffolding (:class:`PoolPressureState`, :class:`PoolPressureMonitor`).
Migration mechanics land in v2 sub-PR 2.
"""

from cachepawl.allocator.avmp.pool import AsymmetricVirtualPool
from cachepawl.allocator.avmp.state import PoolPressureMonitor, PoolPressureState

__all__ = ["AsymmetricVirtualPool", "PoolPressureMonitor", "PoolPressureState"]
