# vLLM Path C Advisory Matrix

This table is advisory/diagnostic evidence only. It does not report runtime
mutation, throughput, serving, or VRAM improvement measurements.

| model | max_model_len | gpu_memory_utilization | max_num_seqs | vanilla_reserved_bytes | vanilla_useful_bytes | cachepawl_proposed_reserved_bytes | estimated_savings_bytes | overestimation_ratio | wasted_fraction | num_blocks | cache_group_count | cache_tensor_count | layer_count | status | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zyphra/Zamba2-2.7B-instruct | 2048 | 0.6 | 1 |  |  |  |  |  |  |  |  |  |  | pending_not_run | matrix point not run |
| Zyphra/Zamba2-2.7B-instruct | 2048 | 0.7 | 1 |  |  |  |  |  |  |  |  |  |  | pending_not_run | matrix point not run |
| Zyphra/Zamba2-2.7B-instruct | 4096 | 0.6 | 1 |  |  |  |  |  |  |  |  |  |  | pending_not_run | matrix point not run |
| Zyphra/Zamba2-2.7B-instruct | 4096 | 0.7 | 1 | 2910781440 | 1679258112 | 1679258112 | 1231523328 | 1.7333734577189286 | 0.4230902777777778 | 329 | 7 | 9 | 63 | completed_existing_baseline |  |
