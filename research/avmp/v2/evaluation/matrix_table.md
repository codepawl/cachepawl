# vLLM Path C Advisory Matrix

This table is advisory/diagnostic evidence only. It does not report runtime
mutation, throughput, serving, or VRAM improvement measurements.

| model | max_model_len | gpu_memory_utilization | max_num_seqs | vanilla_reserved_bytes | vanilla_useful_bytes | cachepawl_proposed_reserved_bytes | estimated_savings_bytes | overestimation_ratio | wasted_fraction | num_blocks | cache_group_count | cache_tensor_count | layer_count | status | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zyphra/Zamba2-2.7B-instruct | 2048 | 0.6 | 1 | 1893335040 | 1092283392 | 1092283392 | 801051648 | 1.7333734577189286 | 0.4230902777777778 | 214 | 7 | 9 | 63 | completed |  |
| Zyphra/Zamba2-2.7B-instruct | 2048 | 0.7 | 1 | 3185049600 | 1837486080 | 1837486080 | 1347563520 | 1.7333734577189286 | 0.4230902777777778 | 360 | 7 | 9 | 63 | completed |  |
| Zyphra/Zamba2-2.7B-instruct | 4096 | 0.6 | 1 | 1619066880 | 934055424 | 934055424 | 685011456 | 1.7333734577189286 | 0.4230902777777778 | 183 | 7 | 9 | 63 | completed |  |
| Zyphra/Zamba2-2.7B-instruct | 4096 | 0.7 | 1 | 2910781440 | 1679258112 | 1679258112 | 1231523328 | 1.7333734577189286 | 0.4230902777777778 | 329 | 7 | 9 | 63 | completed_existing_baseline |  |
