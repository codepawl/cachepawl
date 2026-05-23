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

__all__ = [
    "VllmCacheLayerSpec",
    "VllmCachePlan",
    "VllmIntegrationPlan",
    "get_vllm_version",
    "is_vllm_available",
    "plan_vllm_integration",
    "translate_hybrid_model_spec",
]
