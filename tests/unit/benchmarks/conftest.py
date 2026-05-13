"""Shared fixtures for the benchmark harness unit tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from cachepawl.allocator.base import Allocator, AllocatorStats
from cachepawl.benchmarks.harness.workloads import WorkloadSpec


class FakeAllocator(Allocator):
    """Trivial in-memory allocator used by harness tests.

    Hands out monotonically increasing block ids, recycles freed ids on
    later allocations, and reports plausible occupancy statistics. The
    constructor accepts ``(spec, device)`` to satisfy the
    ``AllocatorFactory`` signature when registered in
    ``cachepawl.benchmarks.REGISTRY``; both are ignored.
    """

    def __init__(
        self,
        spec: WorkloadSpec | None = None,
        device: torch.device | None = None,
        *,
        total_blocks: int = 100_000,
    ) -> None:
        del spec
        del device
        self._total_blocks = total_blocks
        self._next_id = 0
        self._free_pool: list[int] = []
        self._allocated: set[int] = set()
        self.allocate_calls = 0
        self.free_calls = 0

    def allocate(self, num_blocks: int, *, dtype_bytes: int) -> list[int]:
        self.allocate_calls += 1
        del dtype_bytes
        ids: list[int] = []
        for _ in range(num_blocks):
            if self._free_pool:
                bid = self._free_pool.pop()
            else:
                bid = self._next_id
                self._next_id += 1
            self._allocated.add(bid)
            ids.append(bid)
        return ids

    def free(self, block_ids: Sequence[int]) -> None:
        self.free_calls += 1
        for bid in block_ids:
            if bid in self._allocated:
                self._allocated.remove(bid)
                self._free_pool.append(bid)

    def stats(self) -> AllocatorStats:
        allocated = len(self._allocated)
        free = self._total_blocks - allocated
        return AllocatorStats(
            total_blocks=self._total_blocks,
            free_blocks=free,
            allocated_blocks=allocated,
            fragmentation_ratio=0.0,
        )


@pytest.fixture
def fake_allocator() -> FakeAllocator:
    return FakeAllocator()
