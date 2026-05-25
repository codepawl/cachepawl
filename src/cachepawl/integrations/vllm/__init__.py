"""Import-safe vLLM integration planning helpers.

This module intentionally does not import ``vllm`` at module import time.
The runtime shim and allocator replacement are future Sprint 1 work; the
current surface only records the cache-plan shape cachepawl will need.
"""

from cachepawl.integrations.vllm.advisory import (
    MISSING_FIELDS_FOR_MUTATION,
    VllmCacheAdvisoryClassification,
    VllmCacheAdvisoryReport,
    VllmCacheGroupAdvisory,
    advise_vllm_runtime_cache_plan,
)
from cachepawl.integrations.vllm.dry_run import (
    VllmDryRunGroupProposal,
    VllmPlannerDryRunResult,
    dry_run_vllm_planner_probe,
)
from cachepawl.integrations.vllm.observer import (
    RUNTIME_KV_CACHE_CONFIG_PATH,
    VllmRuntimeCacheObservation,
    VllmRuntimeObservationStatus,
    observe_vllm_runtime_cache_plan,
)
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
    "MISSING_FIELDS_FOR_MUTATION",
    "RUNTIME_KV_CACHE_CONFIG_PATH",
    "VllmCacheAdvisoryClassification",
    "VllmCacheAdvisoryReport",
    "VllmCacheGroupAdvisory",
    "VllmCacheLayerSpec",
    "VllmCachePlan",
    "VllmDryRunGroupProposal",
    "VllmIntegrationPlan",
    "VllmPlannerDryRunResult",
    "VllmRuntimeCacheObservation",
    "VllmRuntimeObservationStatus",
    "VllmTranslatedCacheConfig",
    "VllmTranslatedCacheGroup",
    "VllmTranslatedCacheSpec",
    "VllmTranslatedCacheTensor",
    "VllmTranslationError",
    "advise_vllm_runtime_cache_plan",
    "dry_run_vllm_planner_probe",
    "get_vllm_version",
    "is_vllm_available",
    "observe_vllm_runtime_cache_plan",
    "plan_vllm_integration",
    "translate_hybrid_model_spec",
    "translate_kv_cache_config",
    "translate_kv_cache_group",
    "translate_kv_cache_spec",
    "translate_kv_cache_tensor",
]
