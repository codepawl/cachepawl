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
            f"MemoryPool.allocate: implement once the asymmetric two-pool design lands "
            f"(requested num_blocks={num_blocks}, dtype_bytes={dtype_bytes})."
        )

    def free(self, block_ids: Sequence[int]) -> None:
        raise NotImplementedError(
            f"MemoryPool.free: implement once the asymmetric two-pool design lands "
            f"(received {len(block_ids)} block ids)."
        )

    def stats(self) -> AllocatorStats:
        raise NotImplementedError(
            "MemoryPool.stats: implement once the asymmetric two-pool design lands."
        )
