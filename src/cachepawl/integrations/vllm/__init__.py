"""Import-safe vLLM integration planning helpers.

This module intentionally does not import ``vllm`` at module import time.
The runtime shim and allocator replacement are future Sprint 1 work; the
current surface only records the cache-plan shape cachepawl will need.
"""

from cachepawl.integrations.vllm.planning import (
    VllmCacheLayerSpec,
    VllmCachePlan,
    VllmIntegrationPlan,
    get_vllm_version,
    is_vllm_available,
    plan_vllm_integration,
    translate_hybrid_model_spec,
)
from cachepawl.integrations.vllm.translator import (
    VllmTranslatedCacheConfig,
    VllmTranslatedCacheGroup,
    VllmTranslatedCacheSpec,
    VllmTranslatedCacheTensor,
    VllmTranslationError,
    translate_kv_cache_config,
    translate_kv_cache_group,
    translate_kv_cache_spec,
    translate_kv_cache_tensor,
)

__all__ = [
    "VllmCacheLayerSpec",
    "VllmCachePlan",
    "VllmIntegrationPlan",
    "VllmTranslatedCacheConfig",
    "VllmTranslatedCacheGroup",
    "VllmTranslatedCacheSpec",
    "VllmTranslatedCacheTensor",
    "VllmTranslationError",
    "get_vllm_version",
    "is_vllm_available",
    "plan_vllm_integration",
    "translate_hybrid_model_spec",
    "translate_kv_cache_config",
    "translate_kv_cache_group",
    "translate_kv_cache_spec",
    "translate_kv_cache_tensor",
]
