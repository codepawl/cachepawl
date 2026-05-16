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
| padded_unified | jamba_1_5_mini | 1gib | 0.00 +- 0.00 | 0.690 +- 0.006 | 0.995 | 2.76 | 121.90 | 0.7 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.587 +- 0.009 | 0.997 | 2.77 | 102.80 | 0.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 8gib | 0.03 +- 0.00 | 0.488 +- 0.014 | 0.998 | 2.76 | 100.75 | 0.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.696 +- 0.011 | 0.996 | 2.76 | 92.14 | 280.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 0.00 +- 0.00 | 0.701 +- 0.008 | 0.991 | 2.86 | 106.05 | 201.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 8gib | 0.00 +- 0.00 | 0.690 +- 0.006 | 0.995 | 2.83 | 123.81 | 0.7 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.175 +- 0.020 | 0.961 | 3.07 | 1740.97 | 3.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.228 +- 0.005 | 0.984 | 3.19 | 11384.70 | 0.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.352 +- 0.004 | 0.992 | 3.13 | 87.62 | 0.0 | - | 4096.000 | 4096.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.162 +- 0.021 | 0.959 | 3.03 | 1669.01 | 3.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.201 +- 0.006 | 0.981 | 3.19 | 13669.34 | 0.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.305 +- 0.004 | 0.990 | 3.16 | 10246.13 | 0.0 | - | 4096.000 | 4096.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.213 +- 0.010 | 0.969 | 3.31 | 1232.12 | 256.7 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.383 +- 0.008 | 0.981 | 3.27 | 3002.53 | 4.3 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 8gib | 0.04 +- 0.00 | 0.560 +- 0.005 | 0.985 | 3.52 | 6467.02 | 6.0 | - | 819.188 | 7372.750 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.114 +- 0.011 | 0.949 | 3.21 | 342.93 | 256.7 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 0.01 +- 0.00 | 0.186 +- 0.025 | 0.952 | 3.29 | 2124.99 | 4.3 | - | 409.562 | 3686.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 8gib | 0.02 +- 0.00 | 0.183 +- 0.013 | 0.961 | 3.41 | 9818.91 | 6.0 | - | 819.188 | 7372.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.175 +- 0.020 | 0.961 | 5.87 | 298.75 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.228 +- 0.005 | 0.984 | 5.99 | 212.14 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.352 +- 0.004 | 0.992 | 6.61 | 187.59 | 0.0 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.162 +- 0.021 | 0.959 | 6.31 | 343.53 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.201 +- 0.006 | 0.981 | 5.87 | 209.74 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.305 +- 0.004 | 0.990 | 6.10 | 175.49 | 0.0 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.174 +- 0.020 | 0.961 | 8.92 | 357.52 | 4.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.228 +- 0.005 | 0.984 | 7.94 | 238.89 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.352 +- 0.004 | 0.992 | 7.65 | 181.24 | 0.0 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.158 +- 0.015 | 0.959 | 7.17 | 265.95 | 3.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.201 +- 0.006 | 0.981 | 7.88 | 225.37 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.305 +- 0.004 | 0.990 | 7.38 | 177.86 | 0.0 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |

## Workload: mixed_long

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 0.00 +- 0.00 | 0.847 +- 0.026 | 0.999 | 2.49 | 274.98 | 101.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.768 +- 0.057 | 1.000 | 2.49 | 281.80 | 99.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 8gib | 0.03 +- 0.00 | 0.637 +- 0.012 | 1.000 | 3.01 | 328.42 | 92.7 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.819 +- 0.015 | 1.000 | 2.74 | 247.36 | 221.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 0.00 +- 0.00 | 0.866 +- 0.006 | 0.998 | 2.27 | 240.10 | 151.7 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 8gib | 0.00 +- 0.00 | 0.847 +- 0.026 | 0.999 | 2.27 | 259.87 | 101.3 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.310 +- 0.044 | 0.997 | 2.94 | 1986.62 | 101.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.317 +- 0.044 | 0.999 | 2.98 | 656.36 | 92.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.368 +- 0.041 | 0.999 | 3.16 | 651.30 | 81.3 | - | 4096.000 | 4096.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.285 +- 0.051 | 1.000 | 2.98 | 1444.79 | 101.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.267 +- 0.054 | 1.000 | 2.87 | 11703.83 | 92.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.273 +- 0.050 | 1.000 | 2.75 | 28654.29 | 81.3 | - | 4096.000 | 4096.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.476 +- 0.017 | 0.989 | 3.13 | 855.43 | 145.3 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.661 +- 0.051 | 0.997 | 3.06 | 761.88 | 100.7 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 8gib | 0.04 +- 0.00 | 0.735 +- 0.019 | 0.999 | 2.91 | 1700.54 | 97.3 | - | 819.188 | 7372.750 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.507 +- 0.038 | 0.998 | 2.62 | 334.58 | 145.3 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 0.01 +- 0.00 | 0.464 +- 0.127 | 0.999 | 2.80 | 2054.06 | 100.7 | - | 409.562 | 3686.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 8gib | 0.02 +- 0.00 | 0.432 +- 0.047 | 0.996 | 3.23 | 4976.04 | 97.3 | - | 819.188 | 7372.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.310 +- 0.044 | 0.997 | 5.52 | 954.18 | 101.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.317 +- 0.044 | 0.999 | 5.20 | 846.15 | 92.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.368 +- 0.041 | 0.999 | 5.71 | 878.00 | 81.3 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.285 +- 0.051 | 1.000 | 5.64 | 906.83 | 101.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.267 +- 0.054 | 1.000 | 6.03 | 876.83 | 92.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.273 +- 0.050 | 1.000 | 5.76 | 983.38 | 81.3 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.323 +- 0.036 | 0.997 | 6.17 | 885.51 | 102.0 | - | 518.500 | 505.500 | 26 | 6.50 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.320 +- 0.009 | 0.999 | 7.38 | 970.05 | 91.3 | - | 2070.750 | 2025.250 | 91 | 22.75 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.350 +- 0.016 | 0.999 | 7.71 | 851.79 | 84.0 | - | 4115.500 | 4076.500 | 78 | 19.50 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.302 +- 0.039 | 1.000 | 7.11 | 1049.69 | 103.0 | - | 520.000 | 504.000 | 4 | 8.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.250 +- 0.032 | 1.000 | 6.85 | 939.64 | 93.0 | - | 2058.000 | 2038.000 | 5 | 10.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.259 +- 0.029 | 1.000 | 6.97 | 833.93 | 85.3 | - | 4116.000 | 4076.000 | 10 | 20.00 | 0 | 0.00 |

## Workload: agentic_burst

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 0.00 +- 0.00 | 0.881 +- 0.002 | 0.999 | 2.62 | 1147.26 | 31.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.790 +- 0.018 | 1.000 | 2.65 | 1186.92 | 34.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 8gib | 0.03 +- 0.00 | 0.708 +- 0.002 | 1.000 | 2.78 | 1242.42 | 33.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.877 +- 0.046 | 1.000 | 2.70 | 416.62 | 392.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 0.00 +- 0.00 | 0.910 +- 0.032 | 1.000 | 2.32 | 944.72 | 55.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 8gib | 0.00 +- 0.00 | 0.881 +- 0.002 | 0.999 | 2.25 | 1044.78 | 31.0 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.534 +- 0.027 | 0.998 | 2.78 | 3084.82 | 38.0 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.384 +- 0.081 | 0.999 | 2.88 | 15546.65 | 40.7 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.390 +- 0.022 | 0.999 | 2.86 | 29948.23 | 35.0 | - | 4096.000 | 4096.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.551 +- 0.027 | 1.000 | 3.11 | 2853.48 | 38.0 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.345 +- 0.101 | 0.999 | 3.36 | 24859.03 | 40.7 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.298 +- 0.027 | 1.000 | 3.26 | 280754.66 | 35.0 | - | 4096.000 | 4096.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.586 +- 0.025 | 0.995 | 3.24 | 2496.18 | 67.7 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.679 +- 0.005 | 0.998 | 3.30 | 4197.58 | 36.0 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 8gib | 0.04 +- 0.00 | 0.746 +- 0.056 | 0.999 | 3.22 | 7628.97 | 39.7 | - | 819.188 | 7372.750 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.775 +- 0.010 | 0.998 | 2.96 | 1231.67 | 67.7 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 0.01 +- 0.00 | 0.499 +- 0.022 | 0.997 | 3.14 | 3252.76 | 36.0 | - | 409.562 | 3686.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 8gib | 0.02 +- 0.00 | 0.456 +- 0.141 | 0.999 | 3.26 | 8585.09 | 39.7 | - | 819.188 | 7372.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.534 +- 0.027 | 0.998 | 5.79 | 3489.00 | 38.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.384 +- 0.081 | 0.999 | 5.89 | 3274.39 | 40.7 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.390 +- 0.022 | 0.999 | 5.73 | 3169.82 | 35.0 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.551 +- 0.027 | 1.000 | 5.25 | 3367.54 | 38.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.345 +- 0.101 | 0.999 | 5.48 | 3295.47 | 40.7 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.298 +- 0.027 | 1.000 | 6.77 | 4228.83 | 35.0 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.537 +- 0.027 | 0.998 | 6.51 | 3195.37 | 39.0 | - | 514.500 | 509.500 | 10 | 2.50 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.393 +- 0.065 | 0.999 | 7.21 | 3249.46 | 37.7 | - | 2056.750 | 2039.250 | 35 | 8.75 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.380 +- 0.019 | 0.999 | 8.00 | 4167.27 | 32.7 | - | 4104.250 | 4087.750 | 33 | 8.25 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.557 +- 0.028 | 1.000 | 6.73 | 3467.59 | 42.0 | - | 520.000 | 504.000 | 4 | 8.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.351 +- 0.081 | 0.999 | 7.61 | 3179.77 | 37.0 | - | 2054.000 | 2042.000 | 3 | 6.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.300 +- 0.023 | 1.000 | 7.39 | 3286.20 | 32.3 | - | 4104.000 | 4088.000 | 4 | 8.00 | 0 | 0.00 |

## Cross-workload summary

Per-variant aggregate across every workload in this sweep. The v1 baseline AVMP must match is `fixed_dual_mr05`; the v2 target is reducing `total_oom` on the workloads where `fixed_dual_mr09` strands the KV pool, without introducing AVMP-specific OOMs.

| variant | mean_frag_during_load | mean_frag_peak | total_oom | total_kv_free_MiB | total_ssm_free_MiB |
| --- | --- | --- | --- | --- | --- |
| avmp_dynamic_mr05 | 0.319 | 0.992 | 786.3 | 40064.250 | 39807.750 |
| avmp_static_mr05 | 0.319 | 0.992 | 784.7 | 39936.000 | 39936.000 |
| fixed_dual_mr05 | 0.319 | 0.992 | 784.7 | 39936.000 | 39936.000 |
| fixed_dual_mr09 | 0.481 | 0.987 | 1507.3 | 7986.750 | 71875.500 |
| padded_unified | 0.760 | 0.998 | 1826.3 | 0.000 | 0.000 |

## Relative improvement vs padded_unified

### avmp_dynamic_mr05

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +39.09% | -150.90% | -25.81% |
| agentic_burst | jamba_1_5_mini | 4gib | +50.26% | -150.71% | -10.78% |
| agentic_burst | jamba_1_5_mini | 8gib | +46.32% | -150.30% | +1.01% |
| agentic_burst | mamba2_1b3 | 1gib | +36.45% | -1574.22% | +89.29% |
| agentic_burst | mamba2_1b3 | 4gib | +61.39% | -1555.05% | +32.73% |
| agentic_burst | mamba2_1b3 | 8gib | +65.95% | -1552.78% | -4.30% |
| mixed_long | jamba_1_5_mini | 1gib | +61.89% | -151.81% | -0.66% |
| mixed_long | jamba_1_5_mini | 4gib | +58.34% | -151.67% | +7.74% |
| mixed_long | jamba_1_5_mini | 8gib | +44.99% | -150.77% | +9.35% |
| mixed_long | mamba2_1b3 | 1gib | +63.14% | -1574.22% | +53.39% |
| mixed_long | mamba2_1b3 | 4gib | +71.12% | -1560.09% | +38.68% |
| mixed_long | mamba2_1b3 | 8gib | +69.44% | -1559.33% | +15.79% |
| uniform_short | jamba_1_5_mini | 1gib | +74.73% | -150.02% | -500.00% |
| uniform_short | jamba_1_5_mini | 4gib | +61.13% | -150.00% | +100.00% |
| uniform_short | jamba_1_5_mini | 8gib | +27.80% | -150.00% | +0.00% |
| uniform_short | mamba2_1b3 | 1gib | +77.29% | -1552.02% | +98.93% |
| uniform_short | mamba2_1b3 | 4gib | +71.35% | -1550.00% | +100.00% |
| uniform_short | mamba2_1b3 | 8gib | +55.75% | -1550.00% | +100.00% |

### avmp_static_mr05

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +39.39% | -150.00% | -22.58% |
| agentic_burst | jamba_1_5_mini | 4gib | +51.40% | -150.00% | -19.61% |
| agentic_burst | jamba_1_5_mini | 8gib | +44.88% | -150.00% | -6.06% |
| agentic_burst | mamba2_1b3 | 1gib | +37.12% | -1550.00% | +90.31% |
| agentic_burst | mamba2_1b3 | 4gib | +62.11% | -1550.00% | +26.06% |
| agentic_burst | mamba2_1b3 | 8gib | +66.15% | -1550.00% | -12.90% |
| mixed_long | jamba_1_5_mini | 1gib | +63.41% | -150.00% | -0.33% |
| mixed_long | jamba_1_5_mini | 4gib | +58.67% | -150.00% | +7.07% |
| mixed_long | jamba_1_5_mini | 8gib | +42.16% | -150.00% | +12.23% |
| mixed_long | mamba2_1b3 | 1gib | +65.23% | -1550.00% | +54.00% |
| mixed_long | mamba2_1b3 | 4gib | +69.17% | -1550.00% | +39.34% |
| mixed_long | mamba2_1b3 | 8gib | +67.79% | -1550.00% | +19.74% |
| uniform_short | jamba_1_5_mini | 1gib | +74.68% | -150.00% | -450.00% |
| uniform_short | jamba_1_5_mini | 4gib | +61.13% | -150.00% | +100.00% |
| uniform_short | jamba_1_5_mini | 8gib | +27.80% | -150.00% | +0.00% |
| uniform_short | mamba2_1b3 | 1gib | +76.71% | -1550.00% | +98.69% |
| uniform_short | mamba2_1b3 | 4gib | +71.35% | -1550.00% | +100.00% |
| uniform_short | mamba2_1b3 | 8gib | +55.75% | -1550.00% | +100.00% |

### fixed_dual_mr05

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +39.39% | -150.00% | -22.58% |
| agentic_burst | jamba_1_5_mini | 4gib | +51.40% | -150.00% | -19.61% |
| agentic_burst | jamba_1_5_mini | 8gib | +44.88% | -150.00% | -6.06% |
| agentic_burst | mamba2_1b3 | 1gib | +37.12% | -1550.00% | +90.31% |
| agentic_burst | mamba2_1b3 | 4gib | +62.11% | -1550.00% | +26.06% |
| agentic_burst | mamba2_1b3 | 8gib | +66.15% | -1550.00% | -12.90% |
| mixed_long | jamba_1_5_mini | 1gib | +63.41% | -150.00% | -0.33% |
| mixed_long | jamba_1_5_mini | 4gib | +58.67% | -150.00% | +7.07% |
| mixed_long | jamba_1_5_mini | 8gib | +42.16% | -150.00% | +12.23% |
| mixed_long | mamba2_1b3 | 1gib | +65.23% | -1550.00% | +54.00% |
| mixed_long | mamba2_1b3 | 4gib | +69.17% | -1550.00% | +39.34% |
| mixed_long | mamba2_1b3 | 8gib | +67.79% | -1550.00% | +19.74% |
| uniform_short | jamba_1_5_mini | 1gib | +74.68% | -150.00% | -450.00% |
| uniform_short | jamba_1_5_mini | 4gib | +61.13% | -150.00% | +100.00% |
| uniform_short | jamba_1_5_mini | 8gib | +27.80% | -150.00% | +0.00% |
| uniform_short | mamba2_1b3 | 1gib | +76.71% | -1550.00% | +98.69% |
| uniform_short | mamba2_1b3 | 4gib | +71.35% | -1550.00% | +100.00% |
| uniform_short | mamba2_1b3 | 8gib | +55.75% | -1550.00% | +100.00% |

### fixed_dual_mr09

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +33.42% | -29.98% | -118.28% |
| agentic_burst | jamba_1_5_mini | 4gib | +14.11% | -29.99% | -5.88% |
| agentic_burst | jamba_1_5_mini | 8gib | -5.45% | -30.00% | -20.20% |
| agentic_burst | mamba2_1b3 | 1gib | +11.63% | -309.77% | +82.74% |
| agentic_burst | mamba2_1b3 | 4gib | +45.12% | -309.96% | +34.55% |
| agentic_burst | mamba2_1b3 | 8gib | +48.24% | -309.99% | -27.96% |
| mixed_long | jamba_1_5_mini | 1gib | +43.78% | -29.98% | -43.42% |
| mixed_long | jamba_1_5_mini | 4gib | +13.94% | -29.99% | -1.68% |
| mixed_long | jamba_1_5_mini | 8gib | -15.36% | -30.00% | -5.04% |
| mixed_long | mamba2_1b3 | 1gib | +38.10% | -309.77% | +34.24% |
| mixed_long | mamba2_1b3 | 4gib | +46.38% | -309.96% | +33.63% |
| mixed_long | mamba2_1b3 | 8gib | +49.01% | -309.99% | +3.95% |
| uniform_short | jamba_1_5_mini | 1gib | +69.08% | -29.98% | -38400.00% |
| uniform_short | jamba_1_5_mini | 4gib | +34.70% | -29.99% | -1200.00% |
| uniform_short | jamba_1_5_mini | 8gib | -14.87% | -30.00% | -inf |
| uniform_short | mamba2_1b3 | 1gib | +83.62% | -309.77% | +8.44% |
| uniform_short | mamba2_1b3 | 4gib | +73.41% | -309.96% | +97.85% |
| uniform_short | mamba2_1b3 | 8gib | +73.50% | -309.99% | -800.00% |

---

Generated: 2026-05-16 from git SHA 19d190a7d1bb, hardware: cpu (linux x86_64).

Regenerate: `python -m cachepawl.benchmarks.compare --quick --device cpu --output benchmarks/results/baseline/quick/`
