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

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | kv_free_MiB | ssm_free_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 0.00 +- 0.00 | 0.690 +- 0.006 | 0.995 | 1.92 | 93.28 | 0.7 | 0.000 | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.587 +- 0.009 | 0.997 | 2.12 | 89.74 | 0.3 | 0.000 | - | - |
| padded_unified | jamba_1_5_mini | 8gib | 0.03 +- 0.00 | 0.488 +- 0.014 | 0.998 | 2.54 | 78.22 | 0.0 | 0.000 | - | - |
| padded_unified | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.696 +- 0.011 | 0.996 | 1.94 | 71.65 | 280.3 | 0.000 | - | - |
| padded_unified | mamba2_1b3 | 4gib | 0.00 +- 0.00 | 0.701 +- 0.008 | 0.991 | 1.88 | 75.47 | 201.3 | 0.000 | - | - |
| padded_unified | mamba2_1b3 | 8gib | 0.00 +- 0.00 | 0.690 +- 0.006 | 0.995 | 1.77 | 79.87 | 0.7 | 0.000 | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.175 +- 0.020 | 0.961 | 2.46 | 1468.77 | 3.7 | - | 512.000 | 512.000 |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.228 +- 0.005 | 0.984 | 2.89 | 7172.84 | 0.0 | - | 2048.000 | 2048.000 |
| fixed_dual_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.352 +- 0.004 | 0.992 | 2.50 | 74.13 | 0.0 | - | 4096.000 | 4096.000 |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.162 +- 0.021 | 0.959 | 2.52 | 1263.30 | 3.7 | - | 512.000 | 512.000 |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.201 +- 0.006 | 0.981 | 2.36 | 6677.27 | 0.0 | - | 2048.000 | 2048.000 |
| fixed_dual_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.305 +- 0.004 | 0.990 | 2.50 | 4223.34 | 0.0 | - | 4096.000 | 4096.000 |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.213 +- 0.010 | 0.969 | 2.42 | 843.51 | 256.7 | - | 102.375 | 921.500 |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.383 +- 0.008 | 0.981 | 2.33 | 1773.72 | 4.3 | - | 409.562 | 3686.250 |
| fixed_dual_mr09 | jamba_1_5_mini | 8gib | 0.04 +- 0.00 | 0.560 +- 0.005 | 0.985 | 2.31 | 2717.26 | 6.0 | - | 819.188 | 7372.750 |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.114 +- 0.011 | 0.949 | 2.19 | 239.80 | 256.7 | - | 102.375 | 920.000 |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 0.01 +- 0.00 | 0.186 +- 0.025 | 0.952 | 2.34 | 1293.10 | 4.3 | - | 409.562 | 3686.000 |
| fixed_dual_mr09 | mamba2_1b3 | 8gib | 0.02 +- 0.00 | 0.183 +- 0.013 | 0.961 | 2.29 | 3148.12 | 6.0 | - | 819.188 | 7372.000 |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.175 +- 0.020 | 0.961 | 4.28 | 212.34 | 3.7 | - | 512.000 | 512.000 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.228 +- 0.005 | 0.984 | 4.48 | 177.92 | 0.0 | - | 2048.000 | 2048.000 |
| avmp_static_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.352 +- 0.004 | 0.992 | 4.60 | 151.77 | 0.0 | - | 4096.000 | 4096.000 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.162 +- 0.021 | 0.959 | 4.29 | 216.82 | 3.7 | - | 512.000 | 512.000 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.201 +- 0.006 | 0.981 | 5.60 | 210.44 | 0.0 | - | 2048.000 | 2048.000 |
| avmp_static_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.305 +- 0.004 | 0.990 | 4.60 | 148.43 | 0.0 | - | 4096.000 | 4096.000 |

## Workload: mixed_long

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | kv_free_MiB | ssm_free_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 0.00 +- 0.00 | 0.847 +- 0.026 | 0.999 | 2.19 | 231.31 | 101.3 | 0.000 | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.768 +- 0.057 | 1.000 | 1.98 | 231.15 | 99.0 | 0.000 | - | - |
| padded_unified | jamba_1_5_mini | 8gib | 0.03 +- 0.00 | 0.637 +- 0.012 | 1.000 | 2.22 | 242.07 | 92.7 | 0.000 | - | - |
| padded_unified | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.819 +- 0.015 | 1.000 | 1.92 | 206.10 | 221.0 | 0.000 | - | - |
| padded_unified | mamba2_1b3 | 4gib | 0.00 +- 0.00 | 0.866 +- 0.006 | 0.998 | 2.34 | 251.62 | 151.7 | 0.000 | - | - |
| padded_unified | mamba2_1b3 | 8gib | 0.00 +- 0.00 | 0.847 +- 0.026 | 0.999 | 2.19 | 290.29 | 101.3 | 0.000 | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.310 +- 0.044 | 0.997 | 2.98 | 1630.95 | 101.7 | - | 512.000 | 512.000 |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.317 +- 0.044 | 0.999 | 2.57 | 472.34 | 92.0 | - | 2048.000 | 2048.000 |
| fixed_dual_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.368 +- 0.041 | 0.999 | 2.86 | 562.98 | 81.3 | - | 4096.000 | 4096.000 |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.285 +- 0.051 | 1.000 | 2.71 | 1367.15 | 101.7 | - | 512.000 | 512.000 |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.267 +- 0.054 | 1.000 | 2.74 | 9824.90 | 92.0 | - | 2048.000 | 2048.000 |
| fixed_dual_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.273 +- 0.050 | 1.000 | 2.80 | 27832.37 | 81.3 | - | 4096.000 | 4096.000 |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.476 +- 0.017 | 0.989 | 2.52 | 580.17 | 145.3 | - | 102.375 | 921.500 |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.661 +- 0.051 | 0.997 | 2.95 | 578.01 | 100.7 | - | 409.562 | 3686.250 |
| fixed_dual_mr09 | jamba_1_5_mini | 8gib | 0.04 +- 0.00 | 0.735 +- 0.019 | 0.999 | 2.73 | 1142.69 | 97.3 | - | 819.188 | 7372.750 |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.507 +- 0.038 | 0.998 | 2.74 | 292.67 | 145.3 | - | 102.375 | 920.000 |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 0.01 +- 0.00 | 0.464 +- 0.127 | 0.999 | 2.32 | 1211.77 | 100.7 | - | 409.562 | 3686.000 |
| fixed_dual_mr09 | mamba2_1b3 | 8gib | 0.02 +- 0.00 | 0.432 +- 0.047 | 0.996 | 2.36 | 2402.77 | 97.3 | - | 819.188 | 7372.000 |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.310 +- 0.044 | 0.997 | 5.33 | 904.02 | 101.7 | - | 512.000 | 512.000 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.317 +- 0.044 | 0.999 | 4.42 | 765.23 | 92.0 | - | 2048.000 | 2048.000 |
| avmp_static_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.368 +- 0.041 | 0.999 | 4.16 | 762.70 | 81.3 | - | 4096.000 | 4096.000 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.285 +- 0.051 | 1.000 | 4.17 | 736.54 | 101.7 | - | 512.000 | 512.000 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.267 +- 0.054 | 1.000 | 4.38 | 758.10 | 92.0 | - | 2048.000 | 2048.000 |
| avmp_static_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.273 +- 0.050 | 1.000 | 4.27 | 658.77 | 81.3 | - | 4096.000 | 4096.000 |

## Workload: agentic_burst

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | kv_free_MiB | ssm_free_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 0.00 +- 0.00 | 0.881 +- 0.002 | 0.999 | 1.88 | 945.74 | 31.0 | 0.000 | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.790 +- 0.018 | 1.000 | 1.94 | 1032.55 | 34.0 | 0.000 | - | - |
| padded_unified | jamba_1_5_mini | 8gib | 0.03 +- 0.00 | 0.708 +- 0.002 | 1.000 | 2.31 | 1060.82 | 33.0 | 0.000 | - | - |
| padded_unified | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.877 +- 0.046 | 1.000 | 2.08 | 356.41 | 392.0 | 0.000 | - | - |
| padded_unified | mamba2_1b3 | 4gib | 0.00 +- 0.00 | 0.910 +- 0.032 | 1.000 | 1.91 | 869.03 | 55.0 | 0.000 | - | - |
| padded_unified | mamba2_1b3 | 8gib | 0.00 +- 0.00 | 0.881 +- 0.002 | 0.999 | 2.39 | 1039.28 | 31.0 | 0.000 | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.534 +- 0.027 | 0.998 | 2.47 | 2970.15 | 38.0 | - | 512.000 | 512.000 |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.384 +- 0.081 | 0.999 | 2.75 | 14774.65 | 40.7 | - | 2048.000 | 2048.000 |
| fixed_dual_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.390 +- 0.022 | 0.999 | 3.01 | 29499.72 | 35.0 | - | 4096.000 | 4096.000 |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.551 +- 0.027 | 1.000 | 3.15 | 2613.14 | 38.0 | - | 512.000 | 512.000 |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.345 +- 0.101 | 0.999 | 3.09 | 19340.63 | 40.7 | - | 2048.000 | 2048.000 |
| fixed_dual_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.298 +- 0.027 | 1.000 | 2.87 | 237750.39 | 35.0 | - | 4096.000 | 4096.000 |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.586 +- 0.025 | 0.995 | 2.16 | 1309.18 | 67.7 | - | 102.375 | 921.500 |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 0.02 +- 0.00 | 0.679 +- 0.005 | 0.998 | 2.17 | 2128.05 | 36.0 | - | 409.562 | 3686.250 |
| fixed_dual_mr09 | jamba_1_5_mini | 8gib | 0.04 +- 0.00 | 0.746 +- 0.056 | 0.999 | 2.48 | 3457.05 | 39.7 | - | 819.188 | 7372.750 |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 0.00 +- 0.00 | 0.775 +- 0.010 | 0.998 | 2.10 | 863.58 | 67.7 | - | 102.375 | 920.000 |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 0.01 +- 0.00 | 0.499 +- 0.022 | 0.997 | 2.25 | 2069.43 | 36.0 | - | 409.562 | 3686.000 |
| fixed_dual_mr09 | mamba2_1b3 | 8gib | 0.02 +- 0.00 | 0.456 +- 0.141 | 0.999 | 2.25 | 3865.18 | 39.7 | - | 819.188 | 7372.000 |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.534 +- 0.027 | 0.998 | 4.30 | 2831.39 | 38.0 | - | 512.000 | 512.000 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 0.04 +- 0.00 | 0.384 +- 0.081 | 0.999 | 5.28 | 3185.41 | 40.7 | - | 2048.000 | 2048.000 |
| avmp_static_mr05 | jamba_1_5_mini | 8gib | 0.08 +- 0.00 | 0.390 +- 0.022 | 0.999 | 6.23 | 3044.09 | 35.0 | - | 4096.000 | 4096.000 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 0.01 +- 0.00 | 0.551 +- 0.027 | 1.000 | 4.75 | 2932.17 | 38.0 | - | 512.000 | 512.000 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 0.03 +- 0.00 | 0.345 +- 0.101 | 0.999 | 4.77 | 2963.47 | 40.7 | - | 2048.000 | 2048.000 |
| avmp_static_mr05 | mamba2_1b3 | 8gib | 0.06 +- 0.00 | 0.298 +- 0.027 | 1.000 | 4.31 | 2647.00 | 35.0 | - | 4096.000 | 4096.000 |

## Cross-workload summary

Per-variant aggregate across every workload in this sweep. The v1 baseline AVMP must match is `fixed_dual_mr05`; the v2 target is reducing `total_oom` on the workloads where `fixed_dual_mr09` strands the KV pool, without introducing AVMP-specific OOMs.

| variant | mean_frag_during_load | mean_frag_peak | total_oom | total_kv_free_MiB | total_ssm_free_MiB |
| --- | --- | --- | --- | --- | --- |
| avmp_static_mr05 | 0.319 | 0.992 | 784.7 | 39936.000 | 39936.000 |
| fixed_dual_mr05 | 0.319 | 0.992 | 784.7 | 39936.000 | 39936.000 |
| fixed_dual_mr09 | 0.481 | 0.987 | 1507.3 | 7986.750 | 71875.500 |
| padded_unified | 0.760 | 0.998 | 1826.3 | 0.000 | 0.000 |

## Relative improvement vs padded_unified

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

Generated: 2026-05-16 from git SHA afa8ae7854f2, hardware: cpu (linux x86_64).

Regenerate: `python -m cachepawl.benchmarks.compare --quick --device cpu --output benchmarks/results/baseline/quick/`
