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
| padded_unified | jamba_1_5_mini | 1gib | 1024.00 +- 0.00 | 0.000 +- 0.000 | 0.000 | 2.65 | 110.18 | 0.7 | 284.0 | 1134.7 | 0.998 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.31 | 84.09 | 0.3 | 284.0 | 984.1 | 1.000 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 1.90 | 69.54 | 280.3 | 284.0 | 1111.7 | 0.674 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 1.92 | 78.63 | 201.3 | 284.0 | 1085.0 | 0.742 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.66 | 1600.47 | 3.7 | 284.0 | 45.6 | 0.992 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.40 | 8566.14 | 0.0 | 284.0 | 15.8 | 1.000 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.99 | 1571.93 | 3.7 | 284.0 | 48.3 | 0.992 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.68 | 9482.92 | 0.0 | 284.0 | 12.9 | 1.000 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.52 | 827.29 | 256.7 | 284.0 | 65.4 | 0.676 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.98 | 1893.11 | 4.3 | 284.0 | 37.9 | 0.992 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.04 | 326.33 | 256.7 | 284.0 | 403.8 | 0.676 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.71 | 1731.17 | 4.3 | 284.0 | 41.1 | 0.992 | - | 409.562 | 3686.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.600 +- 0.000 | 0.600 | 4.82 | 230.82 | 3.7 | 284.0 | 494.7 | 0.992 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.01 | 233.57 | 0.0 | 284.0 | 210.8 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 4.53 | 226.54 | 3.7 | 284.0 | 564.1 | 0.992 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.86 | 234.31 | 0.0 | 284.0 | 204.8 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.99 | 260.47 | 5.3 | 284.0 | 404.7 | 0.990 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.83 | 195.10 | 0.0 | 284.0 | 221.0 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.16 | 238.68 | 3.7 | 284.0 | 490.2 | 0.992 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 8.38 | 234.10 | 0.0 | 284.0 | 220.1 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |

## Workload: mixed_long

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | effective_batch_p50 | goodput_req_per_s | completion_ratio | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.37 | 266.06 | 101.3 | 132.0 | 288.6 | 0.836 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.06 | 259.33 | 99.0 | 132.0 | 209.7 | 0.844 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 1.80 | 184.28 | 221.0 | 132.0 | 519.6 | 0.660 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 1.85 | 187.57 | 151.7 | 132.0 | 493.2 | 0.801 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.64 | 1635.78 | 101.7 | 132.0 | 36.5 | 0.844 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.54 | 449.95 | 92.0 | 132.0 | 13.6 | 0.863 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.08 | 1159.01 | 101.7 | 132.0 | 42.0 | 0.844 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.28 | 8664.75 | 92.0 | 132.0 | 8.0 | 0.863 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.45 | 623.23 | 145.3 | 132.0 | 108.2 | 0.848 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.04 | 818.83 | 100.7 | 132.0 | 42.8 | 0.852 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.79 | 345.93 | 145.3 | 132.0 | 239.5 | 0.848 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.57 | 1362.16 | 100.7 | 132.0 | 34.5 | 0.852 | - | 409.562 | 3686.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.60 | 1329.97 | 101.7 | 132.0 | 52.8 | 0.844 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.50 | 801.57 | 92.0 | 132.0 | 69.5 | 0.863 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 4.72 | 962.58 | 101.7 | 132.0 | 68.2 | 0.844 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.56 | 833.91 | 92.0 | 132.0 | 68.0 | 0.863 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 7.07 | 991.80 | 99.3 | 132.0 | 50.7 | 0.840 | - | 800.000 | 224.000 | 9 | 288.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.32 | 838.53 | 75.0 | 132.0 | 43.6 | 0.863 | - | 3264.000 | 832.000 | 38 | 1216.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.65 | 988.50 | 100.7 | 132.0 | 59.8 | 0.844 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.35 | 867.94 | 89.3 | 132.0 | 52.3 | 0.867 | - | 2560.000 | 1536.000 | 2 | 512.00 | 0 | 0.00 |

## Workload: agentic_burst

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | effective_batch_p50 | goodput_req_per_s | completion_ratio | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.43 | 1134.90 | 31.0 | 129.0 | 214.7 | 0.902 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.38 | 1085.56 | 34.0 | 129.0 | 187.9 | 0.902 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 1.85 | 323.33 | 392.0 | 129.0 | 413.2 | 0.480 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 1.85 | 861.15 | 55.0 | 129.0 | 286.6 | 0.848 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.71 | 2684.33 | 38.0 | 129.0 | 32.6 | 0.891 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.23 | 14862.72 | 40.7 | 129.0 | 10.5 | 0.883 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.42 | 2136.47 | 38.0 | 129.0 | 37.1 | 0.891 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.64 | 17424.18 | 40.7 | 129.0 | 7.5 | 0.883 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.80 | 1388.02 | 67.7 | 129.0 | 77.0 | 0.887 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.91 | 2943.31 | 36.0 | 129.0 | 33.1 | 0.891 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.67 | 1027.84 | 67.7 | 129.0 | 199.6 | 0.887 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.80 | 2470.30 | 36.0 | 129.0 | 32.2 | 0.891 | - | 409.562 | 3686.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.66 | 3470.70 | 38.0 | 129.0 | 46.2 | 0.891 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.87 | 3508.89 | 40.7 | 129.0 | 43.1 | 0.883 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.23 | 3602.63 | 38.0 | 129.0 | 48.0 | 0.891 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.09 | 3332.72 | 40.7 | 129.0 | 44.6 | 0.883 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.74 | 3318.70 | 38.3 | 129.0 | 48.3 | 0.883 | - | 672.000 | 352.000 | 5 | 160.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.52 | 3604.74 | 24.7 | 129.0 | 46.7 | 0.934 | - | 2944.000 | 1152.000 | 28 | 896.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.47 | 3032.60 | 36.7 | 129.0 | 51.1 | 0.891 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.75 | 2795.44 | 37.0 | 129.0 | 50.0 | 0.895 | - | 2560.000 | 1536.000 | 2 | 512.00 | 0 | 0.00 |

## Cross-workload summary

Per-variant aggregate across every workload in this sweep. The v1 baseline AVMP must match is `fixed_dual_mr05`; the v2 target is reducing `total_oom` on the workloads where `fixed_dual_mr09` strands the KV pool, without introducing AVMP-specific OOMs.

| variant | mean_frag_during_load | mean_frag_peak | total_oom | mean_effective_batch_p50 | mean_goodput_req_per_s | total_kv_free_MiB | total_ssm_free_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| avmp_dynamic_b128 | 0.444 | 0.444 | 510.0 | 181.7 | 144.9 | 18944.000 | 11776.000 |
| avmp_static_mr05 | 0.430 | 0.430 | 552.0 | 181.7 | 159.6 | 15360.000 | 15360.000 |
| fixed_dual_mr05 | 0.500 | 0.500 | 552.0 | 181.7 | 25.9 | 15360.000 | 15360.000 |
| fixed_dual_mr09 | 0.500 | 0.500 | 1221.3 | 181.7 | 109.6 | 3071.625 | 27641.250 |
| padded_unified | 0.433 | 0.433 | 1567.7 | 181.7 | 577.4 | 0.000 | 0.000 |

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

Generated: 2026-05-20 from git SHA d9ebbaf03b73, hardware: cuda (linux x86_64).

Regenerate: `python -m cachepawl.benchmarks.compare --quick --device cpu --output benchmarks/results/baseline/quick/`
