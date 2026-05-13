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
    assert dataclasses.is_dataclass(AllocatorStats)
    field_names = {f.name for f in dataclasses.fields(AllocatorStats)}
    assert field_names == {
        "total_blocks",
        "free_blocks",
        "allocated_blocks",
        "fragmentation_ratio",
    }
    stats = AllocatorStats(
        total_blocks=128,
        free_blocks=120,
        allocated_blocks=8,
        fragmentation_ratio=0.0,
    )
    assert stats.free_blocks == 120


def test_memory_pool_methods_are_unimplemented() -> None:
    pool = MemoryPool()
    with pytest.raises(NotImplementedError, match=r"MemoryPool\.allocate"):
        pool.allocate(1, dtype_bytes=2)
    with pytest.raises(NotImplementedError, match=r"MemoryPool\.free"):
        pool.free([0])
    with pytest.raises(NotImplementedError, match=r"MemoryPool\.stats"):
        pool.stats()
