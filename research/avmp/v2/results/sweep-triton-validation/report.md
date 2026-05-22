# Allocator baseline comparison

## How to read

- `peak_reserved_MiB`: peak `reserved` figure from the allocator over the run (lower is better). On CUDA this is `torch.cuda.max_memory_reserved`; on CPU it is the pool's `total_blocks` count, so CPU rows show small integers.
- `fragmentation_during_load`: mean of `1 - allocated/reserved` across only the ticks where the workload had at least one active request. Filters out the post-teardown sample that the runner emits when all requests have departed (which would force the ratio to 1.0). Lower is better.
- `fragmentation_peak`: max of that same filtered series; worst-case during load.
- `alloc_p50_us` / `alloc_p99_us`: allocate-call latency in microseconds (latency varies across reruns; not deterministic).
- `oom_count`: number of `OutOfMemoryError` raised during the run (lower is better).
- `effective_batch_p50`: median number of concurrent in-flight requests across ticks with at least one active request (higher is better; more parallelism per pool dollar). Derived from `active_requests_samples` filtered to positive entries.
- `goodput_req_per_s`: requests that completed cleanly per wall-clock second (higher is better). Smoke runs with sub-millisecond walls are noisy; treat as illustrative below ~10 ms wall time.
- `completion_ratio`: fraction of submitted requests that completed without ANY OOM during their lifetime AND with a clean free at departure (higher is better; 1.000 means no OOM rejections).
- `padding_waste_MiB` (padded_unified) and `kv_free_MiB`, `ssm_free_MiB` (fixed_dual) are END-OF-RUN snapshots. On a workload where every request departs cleanly, padding_waste drops to 0 and pool_free returns to the pool total; the snapshots then carry little comparison weight. The interesting rigidity signal is the pair `(oom_count, fragmentation_peak)`: lower OOMs at comparable fragmentation means the allocator absorbed more load.
- `mean +- std`: mean across replicates with population standard deviation (ddof=0).

## Workload: uniform_short

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | effective_batch_p50 | goodput_req_per_s | completion_ratio | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 1024.00 +- 0.00 | 0.000 +- 0.000 | 0.000 | 3.00 | 159.22 | 0.7 | 284.0 | 783.3 | 0.998 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.78 | 110.27 | 0.3 | 284.0 | 716.1 | 1.000 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.75 | 100.43 | 280.3 | 284.0 | 752.6 | 0.674 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.84 | 105.32 | 201.3 | 284.0 | 736.5 | 0.742 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.76 | 1808.31 | 3.7 | 284.0 | 41.1 | 0.992 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.96 | 9882.93 | 0.0 | 284.0 | 13.2 | 1.000 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.75 | 1384.13 | 3.7 | 284.0 | 52.8 | 0.992 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.96 | 10975.20 | 0.0 | 284.0 | 11.6 | 1.000 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.61 | 855.34 | 256.7 | 284.0 | 64.0 | 0.676 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.88 | 1991.75 | 4.3 | 284.0 | 40.8 | 0.992 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.90 | 285.93 | 256.7 | 284.0 | 434.2 | 0.676 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.60 | 1689.36 | 4.3 | 284.0 | 41.0 | 0.992 | - | 409.562 | 3686.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.600 +- 0.000 | 0.600 | 4.63 | 225.26 | 3.7 | 284.0 | 543.8 | 0.992 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.27 | 211.20 | 0.0 | 284.0 | 244.2 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.11 | 266.39 | 3.7 | 284.0 | 471.5 | 0.992 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 7.30 | 239.78 | 0.0 | 284.0 | 226.5 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 10.55 | 395.08 | 5.3 | 284.0 | 270.6 | 0.990 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 7.31 | 223.99 | 0.0 | 284.0 | 199.3 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 10.05 | 375.01 | 3.7 | 284.0 | 281.5 | 0.992 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 10.76 | 296.40 | 0.0 | 284.0 | 173.1 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128_triton | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 70.00 | 2953.37 | 5.3 | 284.0 | 59.6 | 0.990 | - | - | - | - | - | - | - |
| avmp_dynamic_b128_triton | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 85.78 | 4359.39 | 0.0 | 284.0 | 42.1 | 1.000 | - | - | - | - | - | - | - |
| avmp_dynamic_b128_triton | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 84.15 | 3685.99 | 3.7 | 284.0 | 51.0 | 0.992 | - | - | - | - | - | - | - |
| avmp_dynamic_b128_triton | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 84.26 | 3965.66 | 0.0 | 284.0 | 44.3 | 1.000 | - | - | - | - | - | - | - |

## Workload: mixed_long

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | effective_batch_p50 | goodput_req_per_s | completion_ratio | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.03 | 526.40 | 101.3 | 132.0 | 130.4 | 0.836 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.96 | 417.89 | 99.0 | 132.0 | 143.8 | 0.844 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.44 | 237.21 | 221.0 | 132.0 | 379.0 | 0.660 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.56 | 267.43 | 151.7 | 132.0 | 350.7 | 0.801 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.06 | 1591.17 | 101.7 | 132.0 | 33.0 | 0.844 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.56 | 455.79 | 92.0 | 132.0 | 13.1 | 0.863 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.76 | 1209.37 | 101.7 | 132.0 | 39.0 | 0.844 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.21 | 13480.95 | 92.0 | 132.0 | 6.4 | 0.863 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.10 | 560.35 | 145.3 | 132.0 | 111.6 | 0.848 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.31 | 575.59 | 100.7 | 132.0 | 48.2 | 0.852 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.24 | 298.67 | 145.3 | 132.0 | 283.1 | 0.848 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.17 | 1155.79 | 100.7 | 132.0 | 40.6 | 0.852 | - | 409.562 | 3686.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 4.84 | 861.65 | 101.7 | 132.0 | 74.8 | 0.844 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.32 | 847.73 | 92.0 | 132.0 | 62.3 | 0.863 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.91 | 1015.76 | 101.7 | 132.0 | 65.5 | 0.844 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.35 | 1165.64 | 92.0 | 132.0 | 44.0 | 0.863 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 10.79 | 1537.57 | 99.3 | 132.0 | 40.9 | 0.840 | - | 800.000 | 224.000 | 9 | 288.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 8.73 | 1342.60 | 75.0 | 132.0 | 29.2 | 0.863 | - | 3264.000 | 832.000 | 38 | 1216.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 8.04 | 1179.77 | 100.7 | 132.0 | 53.2 | 0.844 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 8.79 | 1100.88 | 89.3 | 132.0 | 43.3 | 0.867 | - | 2560.000 | 1536.000 | 2 | 512.00 | 0 | 0.00 |
| avmp_dynamic_b128_triton | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 60.65 | 7276.62 | 99.3 | 132.0 | 14.6 | 0.840 | - | - | - | - | - | - | - |
| avmp_dynamic_b128_triton | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 51.69 | 7382.68 | 75.0 | 132.0 | 13.0 | 0.863 | - | - | - | - | - | - | - |
| avmp_dynamic_b128_triton | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 55.70 | 7393.70 | 100.7 | 132.0 | 12.9 | 0.844 | - | - | - | - | - | - | - |
| avmp_dynamic_b128_triton | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 71.54 | 9906.68 | 89.3 | 132.0 | 10.8 | 0.867 | - | - | - | - | - | - | - |

## Workload: agentic_burst

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | effective_batch_p50 | goodput_req_per_s | completion_ratio | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.60 | 1240.79 | 31.0 | 129.0 | 204.2 | 0.902 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.96 | 1504.86 | 34.0 | 129.0 | 125.3 | 0.902 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.85 | 477.69 | 392.0 | 129.0 | 274.0 | 0.480 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.88 | 1259.38 | 55.0 | 129.0 | 176.4 | 0.848 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.97 | 3960.75 | 38.0 | 129.0 | 23.7 | 0.891 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.13 | 18833.94 | 40.7 | 129.0 | 8.4 | 0.883 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.19 | 2913.00 | 38.0 | 129.0 | 28.3 | 0.891 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.27 | 29014.92 | 40.7 | 129.0 | 4.9 | 0.883 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.67 | 1633.55 | 67.7 | 129.0 | 64.2 | 0.887 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.23 | 2227.21 | 36.0 | 129.0 | 39.1 | 0.891 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.66 | 1003.61 | 67.7 | 129.0 | 241.1 | 0.887 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.46 | 2040.61 | 36.0 | 129.0 | 36.6 | 0.891 | - | 409.562 | 3686.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.52 | 4250.46 | 38.0 | 129.0 | 39.8 | 0.891 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 7.67 | 4181.52 | 40.7 | 129.0 | 31.7 | 0.883 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.49 | 4560.12 | 38.0 | 129.0 | 35.5 | 0.891 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 7.51 | 5034.28 | 40.7 | 129.0 | 32.0 | 0.883 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 8.42 | 5174.49 | 38.3 | 129.0 | 34.2 | 0.883 | - | 672.000 | 352.000 | 5 | 160.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 8.23 | 4005.10 | 24.7 | 129.0 | 35.9 | 0.934 | - | 2944.000 | 1152.000 | 28 | 896.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 8.26 | 4125.13 | 36.7 | 129.0 | 35.9 | 0.891 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 9.61 | 4152.91 | 37.0 | 129.0 | 33.5 | 0.895 | - | 2560.000 | 1536.000 | 2 | 512.00 | 0 | 0.00 |
| avmp_dynamic_b128_triton | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 61.18 | 38873.92 | 38.3 | 129.0 | 7.6 | 0.883 | - | - | - | - | - | - | - |
| avmp_dynamic_b128_triton | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 68.74 | 34765.17 | 24.7 | 129.0 | 8.7 | 0.934 | - | - | - | - | - | - | - |
| avmp_dynamic_b128_triton | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 84.02 | 40469.74 | 36.7 | 129.0 | 7.1 | 0.891 | - | - | - | - | - | - | - |
| avmp_dynamic_b128_triton | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 84.21 | 40607.30 | 37.0 | 129.0 | 7.1 | 0.895 | - | - | - | - | - | - | - |

## Cross-workload summary

Per-variant aggregate across every workload in this sweep. The v1 baseline AVMP must match is `fixed_dual_mr05`; the v2 target is reducing `total_oom` on the workloads where `fixed_dual_mr09` strands the KV pool, without introducing AVMP-specific OOMs.

| variant | mean_frag_during_load | mean_frag_peak | total_oom | mean_effective_batch_p50 | mean_goodput_req_per_s | total_kv_free_MiB | total_ssm_free_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| avmp_dynamic_b128 | 0.444 | 0.444 | 510.0 | 181.7 | 102.6 | 18944.000 | 11776.000 |
| avmp_dynamic_b128_triton | 0.444 | 0.444 | 510.0 | 181.7 | 23.2 | 0.000 | 0.000 |
| avmp_static_mr05 | 0.430 | 0.430 | 552.0 | 181.7 | 156.0 | 15360.000 | 15360.000 |
| fixed_dual_mr05 | 0.500 | 0.500 | 552.0 | 181.7 | 22.9 | 15360.000 | 15360.000 |
| fixed_dual_mr09 | 0.500 | 0.500 | 1221.3 | 181.7 | 120.4 | 3071.625 | 27641.250 |
| padded_unified | 0.433 | 0.433 | 1567.7 | 181.7 | 397.7 | 0.000 | 0.000 |

## Relative improvement vs padded_unified

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

### avmp_dynamic_b128_triton

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

### fixed_dual_mr09

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +0.00% | +0.00% | -118.28% |
| agentic_burst | jamba_1_5_mini | 4gib | +0.00% | +0.00% | -5.88% |
| agentic_burst | mamba2_1b3 | 1gib | +0.00% | +0.00% | +82.74% |
| agentic_burst | mamba2_1b3 | 4gib | +0.00% | +0.00% | +34.55% |
| mixed_long | jamba_1_5_mini | 1gib | +0.00% | +0.00% | -43.42% |
| mixed_long | jamba_1_5_mini | 4gib | +0.00% | +0.00% | -1.68% |
| mixed_long | mamba2_1b3 | 1gib | +0.00% | +0.00% | +34.24% |
| mixed_long | mamba2_1b3 | 4gib | +0.00% | +0.00% | +33.63% |
| uniform_short | jamba_1_5_mini | 1gib | -inf | -400.00% | -38400.00% |
| uniform_short | jamba_1_5_mini | 4gib | +0.00% | +0.00% | -1200.00% |
| uniform_short | mamba2_1b3 | 1gib | +0.00% | +0.00% | +8.44% |
| uniform_short | mamba2_1b3 | 4gib | +0.00% | +0.00% | +97.85% |

---

Generated: 2026-05-22 from git SHA 28a17eb4d755, hardware: cuda (linux x86_64).

Regenerate: `python -m cachepawl.benchmarks.compare --quick --device cpu --output benchmarks/results/baseline/quick/`
