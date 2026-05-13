"""Memory pool stub.

Future home of the asymmetric two-pool allocator that splits the VRAM
budget between KV blocks (variable-size, attention layers) and SSM
state blocks (fixed-size, Mamba layers).
"""

from __future__ import annotations

from collections.abc import Sequence

from cachepawl.allocator.base import Allocator, AllocatorStats


class MemoryPool(Allocator):
    """Placeholder for the asymmetric KV/SSM block pool."""

    def allocate(self, num_blocks: int, *, dtype_bytes: int) -> list[int]:
        raise NotImplementedError(
            "MemoryPool.allocate: implement once the asymmetric two-pool design lands."
        )

    def free(self, block_ids: Sequence[int]) -> None:
        raise NotImplementedError(
            "MemoryPool.free: implement once the asymmetric two-pool design lands."
        )

    def stats(self) -> AllocatorStats:
        raise NotImplementedError(
            "MemoryPool.stats: implement once the asymmetric two-pool design lands."
        )
