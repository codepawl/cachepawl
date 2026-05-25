# vLLM Runtime Cache Diagnostic

Status: `planner_advisory_available`

This artifact is advisory and diagnostic only. It consumes Cachepawl's
translated runtime vLLM cache-plan observation and does not change vLLM
behavior. Scheduler decisions, allocator behavior, worker tensor layout,
serving behavior, Triton kernels, copy kernels, LSDR, and quality evaluation
remain unchanged.

## Classification

observe_only, planner_advisory_available, mutation_required_for_runtime_effect

Cachepawl can recommend from observed planning metadata, but runtime improvement still requires
a future scheduler, allocator, planner, or worker allocation mutation point.

## Key Metrics

- `num_blocks`: 329
- `cache_group_count`: 7
- `cache_tensor_count`: 9
- `layer_count`: 63
- `available_kv_cache_gpu_memory_bytes`: 2915421184
- `observed_reserved_bytes`: 2910781440
- `observed_useful_bytes`: 1679258112
- `cachepawl_recommended_bytes`: 1679258112
- `advisory_savings_bytes`: 1231523328
- `overestimation_ratio`: 1.7333734577189286
- `wasted_fraction`: 0.4230902777777778

## Planner Input Coverage

| field | available |
|---|---:|
| `available_kv_cache_gpu_memory` | true |
| `cache_group_count` | true |
| `cache_tensor_count` | true |
| `layer_count` | true |
| `num_blocks` | true |
| `per_group_block_size` | true |
| `per_group_cache_kind` | true |
| `per_group_page_size_bytes` | true |
| `per_group_useful_bytes` | true |

## Missing For Runtime Mutation

- stable scheduler or planner construction hook
- allocator or KVCacheManager replacement control point
- worker tensor allocation layout control point
- runtime request-to-block assignment control
- Mamba state-index and attention view rewrite contract
