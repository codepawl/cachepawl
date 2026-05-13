"""Smoke test for the allocator submodule public surface."""

from __future__ import annotations

import dataclasses

import pytest

from cachepawl.allocator import (
    Allocator,
    AllocatorStats,
    EvictionPolicy,
    EvictionPolicyBase,
    MemoryPool,
)


def test_public_symbols_exist() -> None:
    assert issubclass(MemoryPool, Allocator)
    assert issubclass(EvictionPolicyBase, object)


def test_eviction_policy_members() -> None:
    assert {p.name for p in EvictionPolicy} == {"LRU", "LFU", "FIFO", "PRIORITY"}


def test_allocator_stats_is_frozen_dataclass() -> None:
    stats = AllocatorStats(
        total_blocks=128,
        free_blocks=120,
        allocated_blocks=8,
        fragmentation_ratio=0.0,
    )
    assert stats.free_blocks == 120
    with pytest.raises(dataclasses.FrozenInstanceError):
        # setattr bypasses static type narrowing while still triggering the
        # frozen-dataclass guard at runtime.
        setattr(stats, "free_blocks", 0)


def test_memory_pool_methods_are_unimplemented() -> None:
    pool = MemoryPool()
    with pytest.raises(NotImplementedError, match="MemoryPool.allocate"):
        pool.allocate(1, dtype_bytes=2)
    with pytest.raises(NotImplementedError, match="MemoryPool.free"):
        pool.free([0])
    with pytest.raises(NotImplementedError, match="MemoryPool.stats"):
        pool.stats()
