# Allocator baseline comparison

## How to read

- `peak_reserved_MiB`: peak `reserved` figure from the allocator over the run (lower is better). On CUDA this is `torch.cuda.max_memory_reserved`; on CPU it is the pool's `total_blocks` count, so CPU rows show small integers.
- `fragmentation_during_load`: mean of `1 - allocated/reserved` across only the ticks where the workload had at least one active request. Filters out the post-teardown sample that the runner emits when all requests have departed (which would force the ratio to 1.0). Lower is better.
- `fragmentation_peak`: max of that same filtered series; worst-case during load.
- `alloc_p50_us` / `alloc_p99_us`: allocate-call latency in microseconds (latency varies across reruns; not deterministic).
- `oom_count`: number of `OutOfMemoryError` raised during the run (lower is better).
- `padding_waste_MiB` (padded_unified) and `kv_free_MiB`, `ssm_free_MiB` (fixed_dual) are END-OF-RUN snapshots. On a workload where every request departs cleanly, padding_waste drops to 0 and pool_free returns to the pool total; the snapshots then carry little comparison weight. The interesting rigidity signal is the pair `(oom_count, fragmentation_peak)`: lower OOMs at comparable fragmentation means the allocator absorbed more load.
- `mean +- std`: mean across replicates with population standard deviation (ddof=0).

## Workload: uniform_short

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 1024.00 +- 0.00 | 0.000 +- 0.000 | 0.000 | 1.75 | 81.99 | 0.7 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 1.86 | 84.16 | 0.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 1.85 | 66.42 | 280.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.20 | 78.66 | 201.3 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.17 | 1359.49 | 3.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.14 | 5868.94 | 0.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.12 | 1218.53 | 3.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.23 | 6190.27 | 0.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.600 +- 0.000 | 0.600 | 4.64 | 234.69 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 4.72 | 179.30 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 4.80 | 225.28 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 4.45 | 169.02 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b1 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.58 | 219.50 | 4.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b1 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.20 | 194.96 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b1 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.72 | 228.25 | 3.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b1 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.00 | 176.70 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b2 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.92 | 232.04 | 4.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b2 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.61 | 171.09 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b2 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.16 | 236.55 | 4.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b2 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.40 | 187.90 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b4 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.83 | 236.45 | 3.3 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b4 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.81 | 179.56 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b4 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.72 | 233.54 | 3.3 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b4 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.83 | 163.42 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b8 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.65 | 247.35 | 4.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b8 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.63 | 185.52 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b8 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.96 | 227.72 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b8 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.06 | 184.17 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b16 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.03 | 250.00 | 4.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b16 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.87 | 181.07 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b16 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.26 | 248.65 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b16 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.74 | 171.43 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b32 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.81 | 225.69 | 4.3 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b32 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.62 | 176.69 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b32 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 7.00 | 249.30 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b32 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.72 | 172.88 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b64 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.08 | 239.80 | 4.3 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b64 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.96 | 192.06 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b64 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.55 | 220.52 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b64 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.02 | 199.42 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.64 | 227.55 | 5.3 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.06 | 202.14 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.52 | 228.10 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.14 | 187.11 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b256 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.90 | 235.41 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b256 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.74 | 178.77 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b256 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.78 | 228.45 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b256 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.43 | 188.21 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |

## Workload: mixed_long

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.11 | 256.11 | 101.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 1.83 | 224.70 | 99.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 1.73 | 183.78 | 221.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 1.75 | 200.01 | 151.7 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.03 | 1351.05 | 101.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.10 | 352.65 | 92.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 1.98 | 993.50 | 101.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.05 | 5802.73 | 92.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 4.52 | 809.51 | 101.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 4.41 | 789.63 | 92.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 4.35 | 848.34 | 101.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 4.70 | 769.93 | 92.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b1 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.55 | 914.81 | 102.0 | - | 518.500 | 505.500 | 26 | 6.50 | 0 | 0.00 |
| avmp_dynamic_b1 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.65 | 849.28 | 91.3 | - | 2070.750 | 2025.250 | 91 | 22.75 | 0 | 0.00 |
| avmp_dynamic_b1 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.36 | 866.65 | 103.0 | - | 520.000 | 504.000 | 4 | 8.00 | 0 | 0.00 |
| avmp_dynamic_b1 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.47 | 790.79 | 93.0 | - | 2058.000 | 2038.000 | 5 | 10.00 | 0 | 0.00 |
| avmp_dynamic_b2 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.49 | 818.75 | 101.3 | - | 525.500 | 498.500 | 27 | 13.50 | 0 | 0.00 |
| avmp_dynamic_b2 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.60 | 797.19 | 92.0 | - | 2094.500 | 2001.500 | 93 | 46.50 | 0 | 0.00 |
| avmp_dynamic_b2 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.51 | 837.09 | 104.3 | - | 572.000 | 452.000 | 15 | 60.00 | 0 | 0.00 |
| avmp_dynamic_b2 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.55 | 839.42 | 92.7 | - | 2068.000 | 2028.000 | 5 | 20.00 | 0 | 0.00 |
| avmp_dynamic_b4 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.54 | 792.70 | 101.0 | - | 537.000 | 487.000 | 25 | 25.00 | 0 | 0.00 |
| avmp_dynamic_b4 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.56 | 803.62 | 90.0 | - | 2136.000 | 1960.000 | 88 | 88.00 | 0 | 0.00 |
| avmp_dynamic_b4 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.55 | 871.45 | 104.0 | - | 576.000 | 448.000 | 8 | 64.00 | 0 | 0.00 |
| avmp_dynamic_b4 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.95 | 784.23 | 91.0 | - | 2080.000 | 2016.000 | 4 | 32.00 | 0 | 0.00 |
| avmp_dynamic_b8 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.53 | 772.35 | 103.3 | - | 562.000 | 462.000 | 25 | 50.00 | 0 | 0.00 |
| avmp_dynamic_b8 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.60 | 748.24 | 90.3 | - | 2222.000 | 1874.000 | 87 | 174.00 | 0 | 0.00 |
| avmp_dynamic_b8 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.46 | 861.52 | 103.7 | - | 576.000 | 448.000 | 4 | 64.00 | 0 | 0.00 |
| avmp_dynamic_b8 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.43 | 839.81 | 92.0 | - | 2128.000 | 1968.000 | 5 | 80.00 | 0 | 0.00 |
| avmp_dynamic_b16 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.57 | 830.95 | 101.0 | - | 604.000 | 420.000 | 23 | 92.00 | 0 | 0.00 |
| avmp_dynamic_b16 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.48 | 785.17 | 92.0 | - | 2392.000 | 1704.000 | 86 | 344.00 | 0 | 0.00 |
| avmp_dynamic_b16 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.49 | 818.45 | 101.3 | - | 576.000 | 448.000 | 2 | 64.00 | 0 | 0.00 |
| avmp_dynamic_b16 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.75 | 729.83 | 93.0 | - | 2144.000 | 1952.000 | 3 | 96.00 | 0 | 0.00 |
| avmp_dynamic_b32 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.95 | 847.99 | 101.3 | - | 648.000 | 376.000 | 17 | 136.00 | 0 | 0.00 |
| avmp_dynamic_b32 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.20 | 891.35 | 83.7 | - | 2608.000 | 1488.000 | 70 | 560.00 | 0 | 0.00 |
| avmp_dynamic_b32 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 7.84 | 1021.18 | 101.3 | - | 576.000 | 448.000 | 1 | 64.00 | 0 | 0.00 |
| avmp_dynamic_b32 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.84 | 929.73 | 86.0 | - | 2240.000 | 1856.000 | 3 | 192.00 | 0 | 0.00 |
| avmp_dynamic_b64 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.57 | 801.02 | 98.7 | - | 688.000 | 336.000 | 11 | 176.00 | 0 | 0.00 |
| avmp_dynamic_b64 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.57 | 782.77 | 85.3 | - | 2928.000 | 1168.000 | 55 | 880.00 | 0 | 0.00 |
| avmp_dynamic_b64 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.59 | 799.62 | 100.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b64 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.57 | 781.60 | 90.3 | - | 2432.000 | 1664.000 | 3 | 384.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.64 | 796.69 | 99.3 | - | 800.000 | 224.000 | 9 | 288.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.71 | 800.05 | 75.0 | - | 3264.000 | 832.000 | 38 | 1216.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.85 | 802.77 | 100.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.78 | 880.90 | 89.3 | - | 2560.000 | 1536.000 | 2 | 512.00 | 0 | 0.00 |
| avmp_dynamic_b256 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.63 | 857.63 | 98.0 | - | 832.000 | 192.000 | 5 | 320.00 | 0 | 0.00 |
| avmp_dynamic_b256 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.57 | 761.53 | 73.0 | - | 3456.000 | 640.000 | 22 | 1408.00 | 0 | 0.00 |
| avmp_dynamic_b256 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.67 | 784.96 | 101.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b256 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.52 | 791.60 | 93.3 | - | 2560.000 | 1536.000 | 1 | 512.00 | 0 | 0.00 |

## Workload: agentic_burst

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 1.88 | 914.90 | 31.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 1.83 | 1097.60 | 34.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 1.77 | 331.66 | 392.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.07 | 926.55 | 55.0 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.24 | 2397.22 | 38.0 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.08 | 8442.25 | 40.7 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.09 | 1844.74 | 38.0 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.13 | 11756.59 | 40.7 | - | 2048.000 | 2048.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 4.24 | 2707.14 | 38.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 4.52 | 2488.15 | 40.7 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 4.36 | 2742.94 | 38.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 4.46 | 2583.40 | 40.7 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b1 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.74 | 2765.08 | 39.0 | - | 514.500 | 509.500 | 10 | 2.50 | 0 | 0.00 |
| avmp_dynamic_b1 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.53 | 2972.32 | 37.7 | - | 2056.750 | 2039.250 | 35 | 8.75 | 0 | 0.00 |
| avmp_dynamic_b1 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.38 | 2816.46 | 42.0 | - | 520.000 | 504.000 | 4 | 8.00 | 0 | 0.00 |
| avmp_dynamic_b1 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.05 | 2812.38 | 37.0 | - | 2054.000 | 2042.000 | 3 | 6.00 | 0 | 0.00 |
| avmp_dynamic_b2 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.76 | 2805.28 | 41.7 | - | 517.000 | 507.000 | 10 | 5.00 | 0 | 0.00 |
| avmp_dynamic_b2 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.64 | 2946.37 | 37.3 | - | 2065.000 | 2031.000 | 34 | 17.00 | 0 | 0.00 |
| avmp_dynamic_b2 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.49 | 2926.57 | 42.0 | - | 520.000 | 504.000 | 2 | 8.00 | 0 | 0.00 |
| avmp_dynamic_b2 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.83 | 2585.66 | 35.0 | - | 2056.000 | 2040.000 | 2 | 8.00 | 0 | 0.00 |
| avmp_dynamic_b4 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.54 | 2988.62 | 40.3 | - | 522.000 | 502.000 | 10 | 10.00 | 0 | 0.00 |
| avmp_dynamic_b4 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.61 | 2788.79 | 41.0 | - | 2093.000 | 2003.000 | 45 | 45.00 | 0 | 0.00 |
| avmp_dynamic_b4 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.67 | 2956.78 | 42.0 | - | 520.000 | 504.000 | 1 | 8.00 | 0 | 0.00 |
| avmp_dynamic_b4 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.68 | 2697.55 | 36.0 | - | 2064.000 | 2032.000 | 2 | 16.00 | 0 | 0.00 |
| avmp_dynamic_b8 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.74 | 2784.94 | 39.7 | - | 532.000 | 492.000 | 10 | 20.00 | 0 | 0.00 |
| avmp_dynamic_b8 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.72 | 2972.18 | 38.0 | - | 2136.000 | 1960.000 | 44 | 88.00 | 0 | 0.00 |
| avmp_dynamic_b8 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.60 | 2901.09 | 38.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b8 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.92 | 2713.30 | 37.7 | - | 2080.000 | 2016.000 | 2 | 32.00 | 0 | 0.00 |
| avmp_dynamic_b16 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.66 | 2765.34 | 39.7 | - | 552.000 | 472.000 | 10 | 40.00 | 0 | 0.00 |
| avmp_dynamic_b16 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.79 | 3055.82 | 35.7 | - | 2224.000 | 1872.000 | 44 | 176.00 | 0 | 0.00 |
| avmp_dynamic_b16 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.80 | 2811.99 | 37.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b16 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.62 | 2730.83 | 41.0 | - | 2080.000 | 2016.000 | 1 | 32.00 | 0 | 0.00 |
| avmp_dynamic_b32 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.98 | 3003.24 | 37.3 | - | 584.000 | 440.000 | 9 | 72.00 | 0 | 0.00 |
| avmp_dynamic_b32 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.64 | 2816.28 | 32.7 | - | 2368.000 | 1728.000 | 40 | 320.00 | 0 | 0.00 |
| avmp_dynamic_b32 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.63 | 2723.25 | 34.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b32 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.83 | 2708.97 | 39.7 | - | 2112.000 | 1984.000 | 1 | 64.00 | 0 | 0.00 |
| avmp_dynamic_b64 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.70 | 2941.13 | 37.0 | - | 608.000 | 416.000 | 6 | 96.00 | 0 | 0.00 |
| avmp_dynamic_b64 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.65 | 2943.20 | 27.7 | - | 2592.000 | 1504.000 | 34 | 544.00 | 0 | 0.00 |
| avmp_dynamic_b64 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.59 | 2736.44 | 36.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b64 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.77 | 2820.39 | 37.0 | - | 2304.000 | 1792.000 | 2 | 256.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.14 | 3067.49 | 38.3 | - | 672.000 | 352.000 | 5 | 160.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.68 | 2832.53 | 24.7 | - | 2944.000 | 1152.000 | 28 | 896.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.54 | 2874.88 | 36.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.98 | 2822.01 | 37.0 | - | 2560.000 | 1536.000 | 2 | 512.00 | 0 | 0.00 |
| avmp_dynamic_b256 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.55 | 2935.66 | 36.0 | - | 704.000 | 320.000 | 3 | 192.00 | 0 | 0.00 |
| avmp_dynamic_b256 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.81 | 2724.72 | 24.7 | - | 3136.000 | 960.000 | 17 | 1088.00 | 0 | 0.00 |
| avmp_dynamic_b256 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.76 | 2897.27 | 38.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b256 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.28 | 2886.02 | 37.3 | - | 2560.000 | 1536.000 | 1 | 512.00 | 0 | 0.00 |

## Cross-workload summary

Per-variant aggregate across every workload in this sweep. The v1 baseline AVMP must match is `fixed_dual_mr05`; the v2 target is reducing `total_oom` on the workloads where `fixed_dual_mr09` strands the KV pool, without introducing AVMP-specific OOMs.

| variant | mean_frag_during_load | mean_frag_peak | total_oom | total_kv_free_MiB | total_ssm_free_MiB |
| --- | --- | --- | --- | --- | --- |
| avmp_dynamic_b1 | 0.444 | 0.444 | 552.0 | 15432.500 | 15287.500 |
| avmp_dynamic_b128 | 0.444 | 0.444 | 510.0 | 18944.000 | 11776.000 |
| avmp_dynamic_b16 | 0.444 | 0.444 | 549.0 | 16204.000 | 14516.000 |
| avmp_dynamic_b2 | 0.444 | 0.444 | 554.3 | 15538.000 | 15182.000 |
| avmp_dynamic_b256 | 0.444 | 0.444 | 509.3 | 19392.000 | 11328.000 |
| avmp_dynamic_b32 | 0.444 | 0.444 | 524.7 | 16768.000 | 13952.000 |
| avmp_dynamic_b4 | 0.444 | 0.444 | 552.0 | 15648.000 | 15072.000 |
| avmp_dynamic_b64 | 0.444 | 0.444 | 521.3 | 17696.000 | 13024.000 |
| avmp_dynamic_b8 | 0.444 | 0.444 | 551.0 | 15868.000 | 14852.000 |
| avmp_static_mr05 | 0.430 | 0.430 | 552.0 | 15360.000 | 15360.000 |
| fixed_dual_mr05 | 0.500 | 0.500 | 552.0 | 15360.000 | 15360.000 |
| padded_unified | 0.433 | 0.433 | 1567.7 | 0.000 | 0.000 |

## Relative improvement vs padded_unified

### avmp_dynamic_b1

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +2.78% | -80.00% | -25.81% |
| agentic_burst | jamba_1_5_mini | 4gib | +44.44% | -80.00% | -10.78% |
| agentic_burst | mamba2_1b3 | 1gib | +2.78% | -80.00% | +89.29% |
| agentic_burst | mamba2_1b3 | 4gib | +44.44% | -80.00% | +32.73% |
| mixed_long | jamba_1_5_mini | 1gib | +2.78% | -80.00% | -0.66% |
| mixed_long | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +7.74% |
| mixed_long | mamba2_1b3 | 1gib | +2.78% | -80.00% | +53.39% |
| mixed_long | mamba2_1b3 | 4gib | +44.44% | -80.00% | +38.68% |
| uniform_short | jamba_1_5_mini | 1gib | -inf | -800.00% | -500.00% |
| uniform_short | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +100.00% |
| uniform_short | mamba2_1b3 | 1gib | +2.78% | -80.00% | +98.93% |
| uniform_short | mamba2_1b3 | 4gib | +44.44% | -80.00% | +100.00% |

### avmp_dynamic_b128

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +2.78% | -80.00% | -23.66% |
| agentic_burst | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +27.45% |
| agentic_burst | mamba2_1b3 | 1gib | +2.78% | -80.00% | +90.65% |
| agentic_burst | mamba2_1b3 | 4gib | +44.44% | -80.00% | +32.73% |
| mixed_long | jamba_1_5_mini | 1gib | +2.78% | -80.00% | +1.97% |
| mixed_long | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +24.24% |
| mixed_long | mamba2_1b3 | 1gib | +2.78% | -80.00% | +54.45% |
| mixed_long | mamba2_1b3 | 4gib | +44.44% | -80.00% | +41.10% |
| uniform_short | jamba_1_5_mini | 1gib | -inf | -800.00% | -700.00% |
| uniform_short | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +100.00% |
| uniform_short | mamba2_1b3 | 1gib | +2.78% | -80.00% | +98.69% |
| uniform_short | mamba2_1b3 | 4gib | +44.44% | -80.00% | +100.00% |

### avmp_dynamic_b16

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +2.78% | -80.00% | -27.96% |
| agentic_burst | jamba_1_5_mini | 4gib | +44.44% | -80.00% | -4.90% |
| agentic_burst | mamba2_1b3 | 1gib | +2.78% | -80.00% | +90.39% |
| agentic_burst | mamba2_1b3 | 4gib | +44.44% | -80.00% | +25.45% |
| mixed_long | jamba_1_5_mini | 1gib | +2.78% | -80.00% | +0.33% |
| mixed_long | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +7.07% |
| mixed_long | mamba2_1b3 | 1gib | +2.78% | -80.00% | +54.15% |
| mixed_long | mamba2_1b3 | 4gib | +44.44% | -80.00% | +38.68% |
| uniform_short | jamba_1_5_mini | 1gib | -inf | -800.00% | -500.00% |
| uniform_short | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +100.00% |
| uniform_short | mamba2_1b3 | 1gib | +2.78% | -80.00% | +98.69% |
| uniform_short | mamba2_1b3 | 4gib | +44.44% | -80.00% | +100.00% |

### avmp_dynamic_b2

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +2.78% | -80.00% | -34.41% |
| agentic_burst | jamba_1_5_mini | 4gib | +44.44% | -80.00% | -9.80% |
| agentic_burst | mamba2_1b3 | 1gib | +2.78% | -80.00% | +89.29% |
| agentic_burst | mamba2_1b3 | 4gib | +44.44% | -80.00% | +36.36% |
| mixed_long | jamba_1_5_mini | 1gib | +2.78% | -80.00% | +0.00% |
| mixed_long | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +7.07% |
| mixed_long | mamba2_1b3 | 1gib | +2.78% | -80.00% | +52.79% |
| mixed_long | mamba2_1b3 | 4gib | +44.44% | -80.00% | +38.90% |
| uniform_short | jamba_1_5_mini | 1gib | -inf | -800.00% | -500.00% |
| uniform_short | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +100.00% |
| uniform_short | mamba2_1b3 | 1gib | +2.78% | -80.00% | +98.57% |
| uniform_short | mamba2_1b3 | 4gib | +44.44% | -80.00% | +100.00% |

### avmp_dynamic_b256

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +2.78% | -80.00% | -16.13% |
| agentic_burst | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +27.45% |
| agentic_burst | mamba2_1b3 | 1gib | +2.78% | -80.00% | +90.31% |
| agentic_burst | mamba2_1b3 | 4gib | +44.44% | -80.00% | +32.12% |
| mixed_long | jamba_1_5_mini | 1gib | +2.78% | -80.00% | +3.29% |
| mixed_long | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +26.26% |
| mixed_long | mamba2_1b3 | 1gib | +2.78% | -80.00% | +54.00% |
| mixed_long | mamba2_1b3 | 4gib | +44.44% | -80.00% | +38.46% |
| uniform_short | jamba_1_5_mini | 1gib | -inf | -800.00% | -450.00% |
| uniform_short | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +100.00% |
| uniform_short | mamba2_1b3 | 1gib | +2.78% | -80.00% | +98.69% |
| uniform_short | mamba2_1b3 | 4gib | +44.44% | -80.00% | +100.00% |

### avmp_dynamic_b32

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +2.78% | -80.00% | -20.43% |
| agentic_burst | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +3.92% |
| agentic_burst | mamba2_1b3 | 1gib | +2.78% | -80.00% | +91.16% |
| agentic_burst | mamba2_1b3 | 4gib | +44.44% | -80.00% | +27.88% |
| mixed_long | jamba_1_5_mini | 1gib | +2.78% | -80.00% | +0.00% |
| mixed_long | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +15.49% |
| mixed_long | mamba2_1b3 | 1gib | +2.78% | -80.00% | +54.15% |
| mixed_long | mamba2_1b3 | 4gib | +44.44% | -80.00% | +43.30% |
| uniform_short | jamba_1_5_mini | 1gib | -inf | -800.00% | -550.00% |
| uniform_short | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +100.00% |
| uniform_short | mamba2_1b3 | 1gib | +2.78% | -80.00% | +98.69% |
| uniform_short | mamba2_1b3 | 4gib | +44.44% | -80.00% | +100.00% |

### avmp_dynamic_b4

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +2.78% | -80.00% | -30.11% |
| agentic_burst | jamba_1_5_mini | 4gib | +44.44% | -80.00% | -20.59% |
| agentic_burst | mamba2_1b3 | 1gib | +2.78% | -80.00% | +89.29% |
| agentic_burst | mamba2_1b3 | 4gib | +44.44% | -80.00% | +34.55% |
| mixed_long | jamba_1_5_mini | 1gib | +2.78% | -80.00% | +0.33% |
| mixed_long | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +9.09% |
| mixed_long | mamba2_1b3 | 1gib | +2.78% | -80.00% | +52.94% |
| mixed_long | mamba2_1b3 | 4gib | +44.44% | -80.00% | +40.00% |
| uniform_short | jamba_1_5_mini | 1gib | -inf | -800.00% | -400.00% |
| uniform_short | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +100.00% |
| uniform_short | mamba2_1b3 | 1gib | +2.78% | -80.00% | +98.81% |
| uniform_short | mamba2_1b3 | 4gib | +44.44% | -80.00% | +100.00% |

### avmp_dynamic_b64

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +2.78% | -80.00% | -19.35% |
| agentic_burst | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +18.63% |
| agentic_burst | mamba2_1b3 | 1gib | +2.78% | -80.00% | +90.65% |
| agentic_burst | mamba2_1b3 | 4gib | +44.44% | -80.00% | +32.73% |
| mixed_long | jamba_1_5_mini | 1gib | +2.78% | -80.00% | +2.63% |
| mixed_long | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +13.80% |
| mixed_long | mamba2_1b3 | 1gib | +2.78% | -80.00% | +54.45% |
| mixed_long | mamba2_1b3 | 4gib | +44.44% | -80.00% | +40.44% |
| uniform_short | jamba_1_5_mini | 1gib | -inf | -800.00% | -550.00% |
| uniform_short | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +100.00% |
| uniform_short | mamba2_1b3 | 1gib | +2.78% | -80.00% | +98.69% |
| uniform_short | mamba2_1b3 | 4gib | +44.44% | -80.00% | +100.00% |

### avmp_dynamic_b8

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +2.78% | -80.00% | -27.96% |
| agentic_burst | jamba_1_5_mini | 4gib | +44.44% | -80.00% | -11.76% |
| agentic_burst | mamba2_1b3 | 1gib | +2.78% | -80.00% | +90.31% |
| agentic_burst | mamba2_1b3 | 4gib | +44.44% | -80.00% | +31.52% |
| mixed_long | jamba_1_5_mini | 1gib | +2.78% | -80.00% | -1.97% |
| mixed_long | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +8.75% |
| mixed_long | mamba2_1b3 | 1gib | +2.78% | -80.00% | +53.09% |
| mixed_long | mamba2_1b3 | 4gib | +44.44% | -80.00% | +39.34% |
| uniform_short | jamba_1_5_mini | 1gib | -inf | -800.00% | -600.00% |
| uniform_short | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +100.00% |
| uniform_short | mamba2_1b3 | 1gib | +2.78% | -80.00% | +98.69% |
| uniform_short | mamba2_1b3 | 4gib | +44.44% | -80.00% | +100.00% |

### avmp_static_mr05

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +2.78% | -80.00% | -22.58% |
| agentic_burst | jamba_1_5_mini | 4gib | +44.44% | -80.00% | -19.61% |
| agentic_burst | mamba2_1b3 | 1gib | +2.78% | -80.00% | +90.31% |
| agentic_burst | mamba2_1b3 | 4gib | +44.44% | -80.00% | +26.06% |
| mixed_long | jamba_1_5_mini | 1gib | +2.78% | -80.00% | -0.33% |
| mixed_long | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +7.07% |
| mixed_long | mamba2_1b3 | 1gib | +2.78% | -80.00% | +54.00% |
| mixed_long | mamba2_1b3 | 4gib | +44.44% | -80.00% | +39.34% |
| uniform_short | jamba_1_5_mini | 1gib | -inf | -400.00% | -450.00% |
| uniform_short | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +100.00% |
| uniform_short | mamba2_1b3 | 1gib | +2.78% | -80.00% | +98.69% |
| uniform_short | mamba2_1b3 | 4gib | +44.44% | -80.00% | +100.00% |

### fixed_dual_mr05

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +0.00% | +0.00% | -22.58% |
| agentic_burst | jamba_1_5_mini | 4gib | +0.00% | +0.00% | -19.61% |
| agentic_burst | mamba2_1b3 | 1gib | +0.00% | +0.00% | +90.31% |
| agentic_burst | mamba2_1b3 | 4gib | +0.00% | +0.00% | +26.06% |
| mixed_long | jamba_1_5_mini | 1gib | +0.00% | +0.00% | -0.33% |
| mixed_long | jamba_1_5_mini | 4gib | +0.00% | +0.00% | +7.07% |
| mixed_long | mamba2_1b3 | 1gib | +0.00% | +0.00% | +54.00% |
| mixed_long | mamba2_1b3 | 4gib | +0.00% | +0.00% | +39.34% |
| uniform_short | jamba_1_5_mini | 1gib | -inf | -400.00% | -450.00% |
| uniform_short | jamba_1_5_mini | 4gib | +0.00% | +0.00% | +100.00% |
| uniform_short | mamba2_1b3 | 1gib | +0.00% | +0.00% | +98.69% |
| uniform_short | mamba2_1b3 | 4gib | +0.00% | +0.00% | +100.00% |

---

Generated: 2026-05-17 from git SHA 7d3b8eb13fbb, hardware: cuda (linux x86_64).

Regenerate: `python -m cachepawl.benchmarks.compare --quick --device cpu --output benchmarks/results/baseline/quick/`
