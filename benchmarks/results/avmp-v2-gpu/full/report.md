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
| padded_unified | jamba_1_5_mini | 1gib | 1024.00 +- 0.00 | 0.000 +- 0.000 | 0.000 | 2.93 | 152.76 | 0.7 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.04 | 159.37 | 0.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.93 | 115.15 | 280.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.96 | 135.06 | 201.3 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.88 | 2005.79 | 3.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.02 | 15953.87 | 0.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.34 | 2347.07 | 3.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.49 | 22259.56 | 0.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.27 | 1249.70 | 256.7 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.18 | 3036.88 | 4.3 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.06 | 320.63 | 256.7 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.23 | 2447.43 | 4.3 | - | 409.562 | 3686.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.600 +- 0.000 | 0.600 | 6.61 | 340.03 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.46 | 273.57 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.97 | 290.77 | 3.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.38 | 249.19 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 8.91 | 426.40 | 4.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 9.01 | 334.88 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 8.43 | 343.22 | 3.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 9.38 | 373.52 | 0.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |

## Workload: mixed_long

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.17 | 473.76 | 101.3 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.22 | 405.16 | 99.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.50 | 223.48 | 221.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.50 | 239.59 | 151.7 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.36 | 2802.37 | 101.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.08 | 647.01 | 92.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.11 | 1549.81 | 101.7 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.29 | 16965.55 | 92.0 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.20 | 899.09 | 145.3 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.18 | 967.69 | 100.7 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.08 | 400.87 | 145.3 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.09 | 2072.19 | 100.7 | - | 409.562 | 3686.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.01 | 1066.80 | 101.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.28 | 1025.15 | 92.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.24 | 1238.72 | 101.7 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.30 | 987.45 | 92.0 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 7.92 | 1205.71 | 102.0 | - | 518.500 | 505.500 | 26 | 6.50 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 8.39 | 1147.25 | 91.3 | - | 2070.750 | 2025.250 | 91 | 22.75 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 7.76 | 1226.70 | 103.0 | - | 520.000 | 504.000 | 4 | 8.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 8.26 | 1186.54 | 93.0 | - | 2058.000 | 2038.000 | 5 | 10.00 | 0 | 0.00 |

## Workload: agentic_burst

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.72 | 1430.06 | 31.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.71 | 1367.08 | 34.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.77 | 468.16 | 392.0 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.62 | 1109.25 | 55.0 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.15 | 4611.91 | 38.0 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.17 | 21037.53 | 40.7 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.04 | 3384.88 | 38.0 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.18 | 27824.30 | 40.7 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.07 | 1958.59 | 67.7 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.21 | 5427.58 | 36.0 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.44 | 1574.11 | 67.7 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 3.12 | 3609.26 | 36.0 | - | 409.562 | 3686.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.19 | 4023.28 | 38.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.34 | 3921.38 | 40.7 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.09 | 3996.47 | 38.0 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 6.48 | 4506.98 | 40.7 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 8.32 | 3718.76 | 39.0 | - | 514.500 | 509.500 | 10 | 2.50 | 0 | 0.00 |
| avmp_dynamic_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 7.62 | 3697.78 | 37.7 | - | 2056.750 | 2039.250 | 35 | 8.75 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 7.77 | 3961.84 | 42.0 | - | 520.000 | 504.000 | 4 | 8.00 | 0 | 0.00 |
| avmp_dynamic_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 7.59 | 3567.32 | 37.0 | - | 2054.000 | 2042.000 | 3 | 6.00 | 0 | 0.00 |

## Cross-workload summary

Per-variant aggregate across every workload in this sweep. The v1 baseline AVMP must match is `fixed_dual_mr05`; the v2 target is reducing `total_oom` on the workloads where `fixed_dual_mr09` strands the KV pool, without introducing AVMP-specific OOMs.

| variant | mean_frag_during_load | mean_frag_peak | total_oom | total_kv_free_MiB | total_ssm_free_MiB |
| --- | --- | --- | --- | --- | --- |
| avmp_dynamic_mr05 | 0.444 | 0.444 | 552.0 | 15432.500 | 15287.500 |
| avmp_static_mr05 | 0.430 | 0.430 | 552.0 | 15360.000 | 15360.000 |
| fixed_dual_mr05 | 0.500 | 0.500 | 552.0 | 15360.000 | 15360.000 |
| fixed_dual_mr09 | 0.500 | 0.500 | 1221.3 | 3071.625 | 27641.250 |
| padded_unified | 0.433 | 0.433 | 1567.7 | 0.000 | 0.000 |

## Relative improvement vs padded_unified

### avmp_dynamic_mr05

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

Generated: 2026-05-16 from git SHA ada718197006, hardware: cuda (linux x86_64).

Regenerate: `python -m cachepawl.benchmarks.compare --quick --device cpu --output benchmarks/results/baseline/quick/`
