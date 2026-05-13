"""Abstract allocator interface."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AllocatorStats:
    """Snapshot of allocator occupancy used by schedulers and tests."""

    total_blocks: int
    free_blocks: int
    allocated_blocks: int
    fragmentation_ratio: float


class Allocator(abc.ABC):
    """Block allocator contract for KV and SSM caches.

    Subclasses own a contiguous block pool and decide how to hand out
    block ids. The contract is intentionally narrow so that asymmetric
    two-pool designs and unified pools can both implement it.
    """

    @abc.abstractmethod
    def allocate(self, num_blocks: int, *, dtype_bytes: int) -> list[int]:
        """Reserve ``num_blocks`` of ``dtype_bytes`` width and return their ids."""

    @abc.abstractmethod
    def free(self, block_ids: Sequence[int]) -> None:
        """Return ``block_ids`` to the pool. Idempotent on already-free ids."""

    @abc.abstractmethod
    def stats(self) -> AllocatorStats:
        """Return a current occupancy snapshot."""
