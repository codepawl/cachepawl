"""Reference baseline allocators used for AVMP comparison.

PaddedUnifiedPool mirrors vLLM's ``HybridKVCacheCoordinator`` page-size
unification; FixedDualPool mirrors SGLang's static dual-pool design.
Each baseline ships with the documented pathology intact so AVMP
comparison data stays meaningful.
"""

from cachepawl.allocator.baselines.common import (
    BackingStore,
    BlockTable,
    CapacityError,
    LayerKindAware,
    LRURequestTracker,
    PageHandle,
    PageTable,
    align_up,
)

__all__ = [
    "BackingStore",
    "BlockTable",
    "CapacityError",
    "LRURequestTracker",
    "LayerKindAware",
    "PageHandle",
    "PageTable",
    "align_up",
]
