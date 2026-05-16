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
| padded_unified | jamba_1_5_mini | 1gib | 0.00 +- 0.00 | 0.690 +- 0.006 | 0.995 | 2.69 | 129.69 | 0.7 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.587 +- 0.009 | 0.997 | 3.02 | 132.69 | 0.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 8gib | 0.03 +- 0.00 | 0.488 +- 0.014 | 0.998 | 2.84 | 102.74 | 0.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.696 +- 0.011 | 0.996 | 2.62 | 93.60 | 280.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 0.00 +- 0.00 | 0.701 +- 0.008 | 0.991 | 2.46 | 90.05 | 201.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 8gib | 0.00 +- 0.00 | 0.690 +- 0.006 | 0.995 | 2.90 | 147.85 | 0.7 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.175 +- 0.020 | 0.961 | 3.11 | 2208.91 | 3.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.228 +- 0.005 | 0.984 | 3.03 | 12546.97 | 0.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.352 +- 0.004 | 0.992 | 2.92 | 78.45 | 0.0 | - | 4096.000 | 4096.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.162 +- 0.021 | 0.959 | 2.95 | 1591.98 | 3.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.201 +- 0.006 | 0.981 | 3.08 | 11792.72 | 0.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.305 +- 0.004 | 0.990 | 3.05 | 7327.33 | 0.0 | - | 4096.000 | 4096.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.213 +- 0.010 | 0.969 | 3.12 | 1026.34 | 256.7 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.383 +- 0.008 | 0.981 | 3.07 | 2860.66 | 4.3 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 8gib | 0.04 +- 0.00 | 0.560 +- 0.005 | 0.985 | 3.31 | 6670.46 | 6.0 | - | 819.188 | 7372.750 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.114 +- 0.011 | 0.949 | 3.09 | 329.08 | 256.7 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 0.01 +- 0.00 | 0.186 +- 0.025 | 0.952 | 3.18 | 1948.87 | 4.3 | - | 409.562 | 3686.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 8gib | 0.02 +- 0.00 | 0.183 +- 0.013 | 0.961 | 3.11 | 5013.71 | 6.0 | - | 819.188 | 7372.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.175 +- 0.020 | 0.961 | 4.49 | 219.65 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.228 +- 0.005 | 0.984 | 5.22 | 186.39 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.352 +- 0.004 | 0.992 | 5.89 | 159.33 | 0.0 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.162 +- 0.021 | 0.959 | 6.16 | 287.88 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.201 +- 0.006 | 0.981 | 5.31 | 194.03 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.305 +- 0.004 | 0.990 | 4.89 | 154.14 | 0.0 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.174 +- 0.021 | 0.961 | 8.21 | 264.38 | 5.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.228 +- 0.005 | 0.984 | 6.29 | 186.66 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.352 +- 0.004 | 0.992 | 6.66 | 156.61 | 0.0 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.161 +- 0.021 | 0.959 | 7.00 | 283.86 | 4.3 | - | 509.438 | 512.000 | 41 | 2.56 | 479 | 2624.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.196 +- 0.004 | 0.979 | 8.30 | 2932.11 | 1.0 | - | 1839.500 | 2048.000 | 3336 | 208.50 | 0 | 213504.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.284 +- 0.002 | 0.989 | 6.85 | 7386.57 | 0.0 | - | 3674.875 | 4096.000 | 6738 | 421.12 | 0 | 431232.00 |

## Workload: mixed_long

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 0.00 +- 0.00 | 0.847 +- 0.026 | 0.999 | 2.73 | 390.57 | 101.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.768 +- 0.057 | 1.000 | 2.32 | 289.93 | 99.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 8gib | 0.03 +- 0.00 | 0.637 +- 0.012 | 1.000 | 2.50 | 287.22 | 92.7 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.819 +- 0.015 | 1.000 | 2.53 | 235.50 | 221.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 0.00 +- 0.00 | 0.866 +- 0.006 | 0.998 | 2.71 | 265.62 | 151.7 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 8gib | 0.00 +- 0.00 | 0.847 +- 0.026 | 0.999 | 2.52 | 277.82 | 101.3 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.310 +- 0.044 | 0.997 | 2.40 | 1547.10 | 101.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.317 +- 0.044 | 0.999 | 2.61 | 455.86 | 92.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.368 +- 0.041 | 0.999 | 2.87 | 549.93 | 81.3 | - | 4096.000 | 4096.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.285 +- 0.051 | 1.000 | 2.81 | 1446.95 | 101.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.267 +- 0.054 | 1.000 | 2.62 | 11070.66 | 92.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.273 +- 0.050 | 1.000 | 2.84 | 28101.18 | 81.3 | - | 4096.000 | 4096.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.476 +- 0.017 | 0.989 | 2.70 | 707.12 | 145.3 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.661 +- 0.051 | 0.997 | 2.52 | 676.52 | 100.7 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 8gib | 0.04 +- 0.00 | 0.735 +- 0.019 | 0.999 | 2.97 | 1137.29 | 97.3 | - | 819.188 | 7372.750 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.507 +- 0.038 | 0.998 | 2.29 | 299.08 | 145.3 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 0.01 +- 0.00 | 0.464 +- 0.127 | 0.999 | 2.79 | 1501.99 | 100.7 | - | 409.562 | 3686.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 8gib | 0.02 +- 0.00 | 0.432 +- 0.047 | 0.996 | 2.75 | 3281.47 | 97.3 | - | 819.188 | 7372.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.310 +- 0.044 | 0.997 | 4.47 | 740.34 | 101.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.317 +- 0.044 | 0.999 | 5.00 | 810.85 | 92.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.368 +- 0.041 | 0.999 | 5.14 | 859.37 | 81.3 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.285 +- 0.051 | 1.000 | 4.48 | 841.95 | 101.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.267 +- 0.054 | 1.000 | 5.35 | 848.25 | 92.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.273 +- 0.050 | 1.000 | 4.72 | 799.15 | 81.3 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.325 +- 0.040 | 0.997 | 6.24 | 992.42 | 103.0 | - | 515.312 | 504.750 | 58 | 10.12 | 1544 | 2560.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.337 +- 0.025 | 0.999 | 5.75 | 1174.93 | 90.0 | - | 2126.250 | 1969.750 | 313 | 78.25 | 3239 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.336 +- 0.018 | 0.999 | 5.96 | 1805.70 | 83.7 | - | 4478.750 | 3713.250 | 1531 | 382.75 | 7711 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.297 +- 0.046 | 1.000 | 6.46 | 1059.30 | 101.7 | - | 510.125 | 508.000 | 61 | 7.75 | 781 | 3904.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.278 +- 0.065 | 1.000 | 6.05 | 760.46 | 91.7 | - | 2068.000 | 2028.000 | 10 | 20.00 | 176 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 8gib | 0.07 +- 0.00 | 0.240 +- 0.032 | 1.000 | 6.04 | 743.73 | 81.7 | - | 4138.000 | 4054.000 | 21 | 42.00 | 348 | 0.00 |

## Workload: agentic_burst

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 0.00 +- 0.00 | 0.881 +- 0.002 | 0.999 | 2.32 | 1146.97 | 31.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.790 +- 0.018 | 1.000 | 2.75 | 1318.13 | 34.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 8gib | 0.03 +- 0.00 | 0.708 +- 0.002 | 1.000 | 2.81 | 1417.51 | 33.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.877 +- 0.046 | 1.000 | 2.80 | 506.93 | 392.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 0.00 +- 0.00 | 0.910 +- 0.032 | 1.000 | 2.67 | 1154.14 | 55.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 8gib | 0.00 +- 0.00 | 0.881 +- 0.002 | 0.999 | 2.42 | 1099.64 | 31.0 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.534 +- 0.027 | 0.998 | 2.88 | 3321.39 | 38.0 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.384 +- 0.081 | 0.999 | 2.82 | 16903.03 | 40.7 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.390 +- 0.022 | 0.999 | 2.71 | 30013.00 | 35.0 | - | 4096.000 | 4096.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.551 +- 0.027 | 1.000 | 2.87 | 2701.10 | 38.0 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.345 +- 0.101 | 0.999 | 2.74 | 19692.43 | 40.7 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.298 +- 0.027 | 1.000 | 3.00 | 235899.44 | 35.0 | - | 4096.000 | 4096.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.586 +- 0.025 | 0.995 | 3.00 | 1564.22 | 67.7 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.679 +- 0.005 | 0.998 | 2.80 | 3018.45 | 36.0 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 8gib | 0.04 +- 0.00 | 0.746 +- 0.056 | 0.999 | 2.82 | 5645.63 | 39.7 | - | 819.188 | 7372.750 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.775 +- 0.010 | 0.998 | 2.84 | 1077.87 | 67.7 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 0.01 +- 0.00 | 0.499 +- 0.022 | 0.997 | 2.93 | 2668.66 | 36.0 | - | 409.562 | 3686.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 8gib | 0.02 +- 0.00 | 0.456 +- 0.141 | 0.999 | 2.48 | 4731.59 | 39.7 | - | 819.188 | 7372.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.534 +- 0.027 | 0.998 | 5.01 | 3145.80 | 38.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.384 +- 0.081 | 0.999 | 5.14 | 3145.78 | 40.7 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.390 +- 0.022 | 0.999 | 5.12 | 2793.06 | 35.0 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.551 +- 0.027 | 1.000 | 4.75 | 2953.44 | 38.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.345 +- 0.101 | 0.999 | 5.21 | 2910.25 | 40.7 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.298 +- 0.027 | 1.000 | 5.44 | 2743.19 | 35.0 | - | 4096.000 | 4096.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.538 +- 0.025 | 0.998 | 6.10 | 2940.78 | 41.0 | - | 516.000 | 504.250 | 86 | 11.50 | 1358 | 2944.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.374 +- 0.092 | 0.999 | 5.98 | 3051.51 | 39.7 | - | 2155.250 | 1940.750 | 429 | 107.25 | 3734 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.357 +- 0.002 | 0.999 | 6.27 | 3907.01 | 34.0 | - | 4529.250 | 3662.750 | 1733 | 433.25 | 6609 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.551 +- 0.030 | 1.000 | 6.49 | 3193.81 | 38.3 | - | 508.625 | 510.000 | 89 | 7.69 | 766 | 5504.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.362 +- 0.077 | 0.999 | 5.83 | 2679.25 | 40.7 | - | 2086.000 | 2010.000 | 19 | 38.00 | 207 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 8gib | 0.07 +- 0.00 | 0.288 +- 0.018 | 1.000 | 6.03 | 2665.50 | 33.0 | - | 4202.000 | 3990.000 | 53 | 106.00 | 521 | 0.00 |

## Cross-workload summary

Per-variant aggregate across every workload in this sweep. The v1 baseline AVMP must match is `fixed_dual_mr05`; the v2 target is reducing `total_oom` on the workloads where `fixed_dual_mr09` strands the KV pool, without introducing AVMP-specific OOMs.

| variant | mean_frag_during_load | mean_frag_peak | total_oom | total_kv_free_MiB | total_ssm_free_MiB |
| --- | --- | --- | --- | --- | --- |
| avmp_dynamic_mr05 | 0.315 | 0.992 | 789.3 | 40513.375 | 38707.500 |
| avmp_static_mr05 | 0.319 | 0.992 | 784.7 | 39936.000 | 39936.000 |
| fixed_dual_mr05 | 0.319 | 0.992 | 784.7 | 39936.000 | 39936.000 |
| fixed_dual_mr09 | 0.481 | 0.987 | 1507.3 | 7986.750 | 71875.500 |
| padded_unified | 0.760 | 0.998 | 1826.3 | 0.000 | 0.000 |

## Relative improvement vs padded_unified

### avmp_dynamic_mr05

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| agentic_burst | jamba_1_5_mini | 1gib | +38.97% | -151.07% | -32.26% |
| agentic_burst | jamba_1_5_mini | 4gib | +52.65% | -158.04% | -16.67% |
| agentic_burst | jamba_1_5_mini | 8gib | +49.59% | -166.11% | -3.03% |
| agentic_burst | mamba2_1b3 | 1gib | +37.12% | -1542.97% | +90.22% |
| agentic_burst | mamba2_1b3 | 4gib | +60.18% | -1580.27% | +26.06% |
| agentic_burst | mamba2_1b3 | 8gib | +67.30% | -1589.86% | -6.45% |
| mixed_long | jamba_1_5_mini | 1gib | +61.63% | -151.16% | -1.64% |
| mixed_long | jamba_1_5_mini | 4gib | +56.16% | -155.75% | +9.09% |
| mixed_long | jamba_1_5_mini | 8gib | +47.19% | -163.42% | +9.71% |
| mixed_long | mamba2_1b3 | 1gib | +63.80% | -1548.18% | +54.00% |
| mixed_long | mamba2_1b3 | 4gib | +67.88% | -1562.61% | +39.56% |
| mixed_long | mamba2_1b3 | 8gib | +71.60% | -1566.90% | +19.41% |
| uniform_short | jamba_1_5_mini | 1gib | +74.82% | -150.07% | -750.00% |
| uniform_short | jamba_1_5_mini | 4gib | +61.13% | -150.00% | +100.00% |
| uniform_short | jamba_1_5_mini | 8gib | +27.78% | -150.04% | +0.00% |
| uniform_short | mamba2_1b3 | 1gib | +76.80% | -1540.69% | +98.45% |
| uniform_short | mamba2_1b3 | 4gib | +72.07% | -1388.12% | +99.50% |
| uniform_short | mamba2_1b3 | 8gib | +58.93% | -1384.61% | +100.00% |

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

Generated: 2026-05-16 from git SHA 1f5b2f73397b, hardware: cpu (linux x86_64).

Regenerate: `python -m cachepawl.benchmarks.compare --quick --device cpu --output benchmarks/results/baseline/quick/`
