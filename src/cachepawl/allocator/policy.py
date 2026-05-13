"""Eviction policy enum and base class."""

from __future__ import annotations

import abc
import enum


class EvictionPolicy(enum.Enum):
    """Selectable eviction policy for the block pool."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    PRIORITY = "priority"


class EvictionPolicyBase(abc.ABC):
    """Pluggable eviction strategy that ranks blocks for reclamation."""

    @abc.abstractmethod
    def record_use(self, block_id: int) -> None:
        """Record that ``block_id`` was just accessed."""

    @abc.abstractmethod
    def select_victims(self, n: int) -> list[int]:
        """Return up to ``n`` block ids that should be evicted next."""
