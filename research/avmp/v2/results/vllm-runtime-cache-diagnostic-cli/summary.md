# vLLM Runtime Cache Diagnostic

Status: `planner_advisory_available`

This diagnostic is advisory-only. It reads translated Cachepawl runtime
observation artifacts and does not rerun vLLM, load a model, mutate runtime
state, replace allocators, or change vLLM behavior.

Runtime savings require a future mutation hook. This command only reports the
observed vLLM cache reservation and the Cachepawl advisory planner view.

## Classification

observe_only, planner_advisory_available, mutation_required_for_runtime_effect

## Key Metrics

- `observed_reserved_bytes`: 2910781440
- `observed_useful_bytes`: 1679258112
- `cachepawl_recommended_bytes`: 1679258112
- `advisory_savings_bytes`: 1231523328
- `overestimation_ratio`: 1.7333734577189286
- `wasted_fraction`: 0.4230902777777778
- `cache_group_count`: 7
- `cache_tensor_count`: 9
- `layer_count`: 63
- `num_blocks`: 329
- `classification`: planner_advisory_available

## Runtime Safety

- `advisory_only`: true
- `vllm_behavior_changed`: false
- `runtime_mutation`: false
- `allocator_replacement`: false
- `runtime_savings_require_future_mutation_hook`: true
- `vllm_required`: false
- `gpu_required`: false
- `nvml_required`: false

## Missing Mutation Fields

- stable scheduler or planner construction hook
- allocator or KVCacheManager replacement control point
- worker tensor allocation layout control point
- runtime request-to-block assignment control
- Mamba state-index and attention view rewrite contract
