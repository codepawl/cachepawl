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
| padded_unified | jamba_1_5_mini | 1gib | 1024.00 +- 0.00 | 0.000 +- 0.000 | 0.000 | 2.32 | 103.53 | 0.7 | 284.0 | 1177.6 | 0.998 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.38 | 90.40 | 0.3 | 284.0 | 940.3 | 1.000 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.13 | 71.50 | 280.3 | 284.0 | 999.9 | 0.674 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 1.88 | 75.66 | 201.3 | 284.0 | 1096.9 | 0.742 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.72 | 1644.16 | 3.7 | 284.0 | 45.7 | 0.992 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.52 | 8304.78 | 0.0 | 284.0 | 14.7 | 1.000 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.46 | 1322.28 | 3.7 | 284.0 | 55.6 | 0.992 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.45 | 6488.11 | 0.0 | 284.0 | 14.6 | 1.000 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.30 | 737.30 | 256.7 | 284.0 | 74.1 | 0.676 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.25 | 1642.23 | 4.3 | 284.0 | 48.7 | 0.992 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.12 | 228.59 | 256.7 | 284.0 | 555.9 | 0.676 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.41 | 1266.08 | 4.3 | 284.0 | 50.8 | 0.992 | - | 409.562 | 3686.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.600 +- 0.000 | 0.600 | 4.41 | 215.20 | 3.7 | 284.0 | 576.1 | 0.992 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 4.45 | 178.89 | 0.0 | 284.0 | 331.0 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 4.70 | 229.92 | 3.7 | 284.0 | 535.2 | 0.992 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 4.29 | 162.09 | 0.0 | 284.0 | 384.0 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.74 | 221.36 | 5.3 | 284.0 | 549.9 | 0.990 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.07 | 192.90 | 0.0 | 284.0 | 293.7 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.45 | 211.42 | 3.7 | 284.0 | 578.6 | 0.992 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.71 | 186.81 | 0.0 | 284.0 | 314.7 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |

## Workload: mixed_long

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | effective_batch_p50 | goodput_req_per_s | completion_ratio | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 1.91 | 202.09 | 101.3 | 132.0 | 347.1 | 0.836 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.76 | 302.11 | 99.0 | 132.0 | 168.5 | 0.844 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.43 | 203.19 | 221.0 | 132.0 | 469.2 | 0.660 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 1.80 | 176.96 | 151.7 | 132.0 | 532.5 | 0.801 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.05 | 1287.98 | 101.7 | 132.0 | 44.2 | 0.844 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.04 | 366.62 | 92.0 | 132.0 | 16.3 | 0.863 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.38 | 1363.70 | 101.7 | 132.0 | 39.0 | 0.844 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.13 | 5805.28 | 92.0 | 132.0 | 9.5 | 0.863 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.05 | 520.84 | 145.3 | 132.0 | 130.5 | 0.848 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.09 | 617.39 | 100.7 | 132.0 | 50.8 | 0.852 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 1.99 | 230.19 | 145.3 | 132.0 | 372.0 | 0.848 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.15 | 1015.32 | 100.7 | 132.0 | 43.8 | 0.852 | - | 409.562 | 3686.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 4.00 | 748.12 | 101.7 | 132.0 | 89.1 | 0.844 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 4.31 | 725.17 | 92.0 | 132.0 | 74.6 | 0.863 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 4.19 | 823.52 | 101.7 | 132.0 | 89.8 | 0.844 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 4.27 | 732.49 | 92.0 | 132.0 | 80.6 | 0.863 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.68 | 783.37 | 99.3 | 132.0 | 72.9 | 0.840 | - | 800.000 | 224.000 | 9 | 288.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.18 | 810.58 | 75.0 | 132.0 | 56.1 | 0.863 | - | 3264.000 | 832.000 | 38 | 1216.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.18 | 877.89 | 100.7 | 132.0 | 70.7 | 0.844 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.25 | 769.54 | 89.3 | 132.0 | 60.6 | 0.867 | - | 2560.000 | 1536.000 | 2 | 512.00 | 0 | 0.00 |

## Workload: agentic_burst

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | effective_batch_p50 | goodput_req_per_s | completion_ratio | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 1.83 | 872.00 | 31.0 | 129.0 | 268.2 | 0.902 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.30 | 1135.85 | 34.0 | 129.0 | 173.1 | 0.902 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.13 | 360.11 | 392.0 | 129.0 | 409.6 | 0.480 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.15 | 972.29 | 55.0 | 129.0 | 263.3 | 0.848 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.19 | 2200.89 | 38.0 | 129.0 | 38.2 | 0.891 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.19 | 8588.63 | 40.7 | 129.0 | 13.8 | 0.883 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.20 | 1884.74 | 38.0 | 129.0 | 42.2 | 0.891 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.19 | 12454.03 | 40.7 | 129.0 | 8.5 | 0.883 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.09 | 1310.93 | 67.7 | 129.0 | 85.5 | 0.887 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.12 | 2110.82 | 36.0 | 129.0 | 43.0 | 0.891 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.15 | 907.36 | 67.7 | 129.0 | 243.0 | 0.887 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.10 | 1873.13 | 36.0 | 129.0 | 40.3 | 0.891 | - | 409.562 | 3686.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 4.54 | 2902.59 | 38.0 | 129.0 | 60.0 | 0.891 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 4.14 | 2679.23 | 40.7 | 129.0 | 56.5 | 0.883 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 4.72 | 2698.36 | 38.0 | 129.0 | 58.7 | 0.891 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 4.27 | 2733.32 | 40.7 | 129.0 | 56.6 | 0.883 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.47 | 3425.02 | 38.3 | 129.0 | 47.9 | 0.883 | - | 672.000 | 352.000 | 5 | 160.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 7.00 | 3425.65 | 24.7 | 129.0 | 44.6 | 0.934 | - | 2944.000 | 1152.000 | 28 | 896.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.72 | 3266.86 | 36.7 | 129.0 | 49.6 | 0.891 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 7.50 | 3141.53 | 37.0 | 129.0 | 45.4 | 0.895 | - | 2560.000 | 1536.000 | 2 | 512.00 | 0 | 0.00 |

## Cross-workload summary

Per-variant aggregate across every workload in this sweep. The v1 baseline AVMP must match is `fixed_dual_mr05`; the v2 target is reducing `total_oom` on the workloads where `fixed_dual_mr09` strands the KV pool, without introducing AVMP-specific OOMs.

| variant | mean_frag_during_load | mean_frag_peak | total_oom | mean_effective_batch_p50 | mean_goodput_req_per_s | total_kv_free_MiB | total_ssm_free_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| avmp_dynamic_b128 | 0.444 | 0.444 | 510.0 | 181.7 | 182.1 | 18944.000 | 11776.000 |
| avmp_static_mr05 | 0.430 | 0.430 | 552.0 | 181.7 | 199.4 | 15360.000 | 15360.000 |
| fixed_dual_mr05 | 0.500 | 0.500 | 552.0 | 181.7 | 28.5 | 15360.000 | 15360.000 |
| fixed_dual_mr09 | 0.500 | 0.500 | 1221.3 | 181.7 | 144.9 | 3071.625 | 27641.250 |
| padded_unified | 0.433 | 0.433 | 1567.7 | 181.7 | 570.5 | 0.000 | 0.000 |

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

Generated: 2026-05-18 from git SHA ecf3d8255509, hardware: cuda (linux x86_64).

Regenerate: `python -m cachepawl.benchmarks.compare --quick --device cpu --output benchmarks/results/baseline/quick/`
