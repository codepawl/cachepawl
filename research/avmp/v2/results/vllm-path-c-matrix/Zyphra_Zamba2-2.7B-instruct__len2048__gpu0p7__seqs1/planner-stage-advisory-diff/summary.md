# vLLM Planner-Stage Advisory Diff Summary

Status: `planner_stage_advisory_diff_available`

This artifact is a planner-stage post-call advisory diff. It consumes the
translated vanilla T002 planner-stage output, computes a Cachepawl proposed
planner result beside it, and does not return that proposal to vLLM.

No vLLM source edits, monkeypatching, allocator replacement, scheduler
mutation, worker layout mutation, long-lived serving, Triton kernels, copy
kernels, LSDR, serving changes, or quality evaluation were performed.

## Key Metrics

- `vanilla_reserved_bytes`: 3185049600
- `vanilla_useful_bytes`: 1837486080
- `cachepawl_proposed_reserved_bytes`: 1837486080
- `estimated_savings_bytes`: 1347563520
- `overestimation_ratio`: 1.7333734577189286
- `wasted_fraction`: 0.4230902777777778
- `cache_group_count`: 7
- `cache_tensor_count`: 9
- `layer_count`: 63
- `num_blocks`: 360
- `non_mutating`: true
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

## Missing Fields That Still Prevent Mutation

- stable scheduler or planner construction hook
- allocator or KVCacheManager replacement control point
- worker tensor allocation layout control point
- runtime request-to-block assignment control
- Mamba state-index and attention view rewrite contract
