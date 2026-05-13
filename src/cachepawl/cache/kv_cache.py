"""KV cache block descriptor and manager interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass

from cachepawl.quant.dtypes import DType


@dataclass(frozen=True, slots=True)
class KVCacheBlock:
    """One KV cache block tied to a layer and dtype."""

    block_id: int
    layer_idx: int
    num_tokens: int
    dtype: DType


class KVCacheManager(abc.ABC):
    """Per-sequence KV cache reservation and release surface."""

    @abc.abstractmethod
    def reserve(self, seq_id: int, num_tokens: int) -> list[KVCacheBlock]:
        """Reserve enough blocks to back ``num_tokens`` for ``seq_id``."""

    @abc.abstractmethod
    def release(self, seq_id: int) -> None:
        """Release every block currently held by ``seq_id``."""

    @abc.abstractmethod
    def get_blocks(self, seq_id: int) -> list[KVCacheBlock]:
        """Return the blocks currently held by ``seq_id`` in layer-major order."""
