"""Reference baseline allocators used for AVMP comparison.

PaddedUnifiedPool mirrors vLLM's ``HybridKVCacheCoordinator`` page-size
unification; FixedDualPool mirrors SGLang's static dual-pool design.
Each baseline ships with the documented pathology intact so AVMP
comparison data stays meaningful.
"""

from cachepawl.allocator.baselines.common import (
    AllocatorContext,
    BackingStore,
    BlockTable,
    CapacityError,
    LRURequestTracker,
    PageHandle,
    PageTable,
    align_up,
)
from cachepawl.allocator.baselines.fixed_dual import FixedDualPool
from cachepawl.allocator.baselines.padded_unified import PaddedUnifiedPool

__all__ = [
    "AllocatorContext",
    "BackingStore",
    "BlockTable",
    "CapacityError",
    "FixedDualPool",
    "LRURequestTracker",
    "PaddedUnifiedPool",
    "PageHandle",
    "PageTable",
    "align_up",
]
