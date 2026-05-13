"""SSM state cache block descriptor and manager interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass

from cachepawl.quant.dtypes import DType


@dataclass(frozen=True, slots=True)
class SSMStateBlock:
    """One SSM state block tied to a layer, state dim, and dtype."""

    block_id: int
    layer_idx: int
    state_dim: int
    dtype: DType


class StateCacheManager(abc.ABC):
    """Per-sequence SSM state reservation and update surface."""

    @abc.abstractmethod
    def reserve(self, seq_id: int) -> SSMStateBlock:
        """Reserve a fresh state block for ``seq_id``."""

    @abc.abstractmethod
    def update(self, seq_id: int, state: SSMStateBlock) -> None:
        """Replace the state block held by ``seq_id`` with ``state``."""

    @abc.abstractmethod
    def release(self, seq_id: int) -> None:
        """Release the state block currently held by ``seq_id``."""
