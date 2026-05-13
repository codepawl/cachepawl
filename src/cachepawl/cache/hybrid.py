"""Coordinator that wires the KV and SSM managers together."""

from __future__ import annotations

import abc

from cachepawl.cache.kv_cache import KVCacheManager
from cachepawl.cache.state_cache import StateCacheManager


class HybridCacheCoordinator(abc.ABC):
    """Route per-layer steps to the right cache manager.

    A hybrid model interleaves attention and SSM layers. The
    coordinator owns one of each manager and dispatches ``step`` calls
    based on the layer kind tracked by the model spec.
    """

    @abc.abstractmethod
    def kv_manager(self) -> KVCacheManager:
        """Return the KV cache manager for attention layers."""

    @abc.abstractmethod
    def state_manager(self) -> StateCacheManager:
        """Return the state cache manager for SSM layers."""

    @abc.abstractmethod
    def step(self, seq_id: int, layer_idx: int) -> None:
        """Advance the cache state for ``seq_id`` at ``layer_idx``."""
