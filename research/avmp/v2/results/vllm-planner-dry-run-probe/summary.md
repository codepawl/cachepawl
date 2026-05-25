# vLLM Planner Dry-Run Probe Summary

Status: `planner_dry_run_available`

This artifact is a planner-level dry run. It consumes the translated vanilla
vLLM cache-plan observation, computes a Cachepawl proposed planner view beside
the vanilla plan, and does not return that proposal to vLLM.

No vLLM source edits, monkeypatching, allocator replacement, scheduler
mutation, worker layout mutation, long-lived serving, Triton kernels, copy
kernels, LSDR, serving changes, or quality evaluation were performed.

## Key Metrics

- `vanilla_observed_reserved_bytes`: 2910781440
- `vanilla_observed_useful_bytes`: 1679258112
- `cachepawl_proposed_reserved_bytes`: 1679258112
- `estimated_savings_bytes`: 1231523328
- `overestimation_ratio`: 1.7333734577189286
- `wasted_fraction`: 0.4230902777777778
- `safe_for_advisory_only`: true
- `returned_to_vllm`: false
- `vllm_behavior_changed`: false

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

## Missing For Mutation

- stable scheduler or planner construction hook
- allocator or KVCacheManager replacement control point
- worker tensor allocation layout control point
- runtime request-to-block assignment control
- Mamba state-index and attention view rewrite contract
