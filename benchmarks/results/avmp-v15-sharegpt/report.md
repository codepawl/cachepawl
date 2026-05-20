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

## Workload: sharegpt_replay

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_during_load | fragmentation_peak | alloc_p50_us | alloc_p99_us | oom_count | effective_batch_p50 | goodput_req_per_s | completion_ratio | padding_waste_MiB | kv_free_MiB | ssm_free_MiB | rebalance_count | bytes_migrated_MiB | throttle_skips | waste_KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 1024.00 +- 0.00 | 0.000 +- 0.000 | 0.000 | 1.98 | 29.98 | 27.0 | 155.0 | 1453.0 | 0.967 | 0.000 | - | - | - | - | - | - |
| padded_unified | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.88 | 30.58 | 1.3 | 155.0 | 639.7 | 0.996 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.72 | 28.78 | 220.7 | 155.0 | 1203.1 | 0.805 | 0.000 | - | - | - | - | - | - |
| padded_unified | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.98 | 43.11 | 64.7 | 155.0 | 1136.5 | 0.936 | 0.000 | - | - | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.67 | 955.18 | 4.7 | 155.0 | 110.1 | 0.994 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.75 | 23.15 | 0.0 | 155.0 | 680.6 | 1.000 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.38 | 880.79 | 4.7 | 155.0 | 121.9 | 0.994 | - | 512.000 | 512.000 | - | - | - | - |
| fixed_dual_mr05 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.65 | 943.82 | 0.0 | 155.0 | 71.1 | 1.000 | - | 2048.000 | 2048.000 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 2.65 | 731.49 | 123.3 | 155.0 | 134.2 | 0.893 | - | 102.375 | 921.500 | - | - | - | - |
| fixed_dual_mr09 | jamba_1_5_mini | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.55 | 29.23 | 12.3 | 155.0 | 248.4 | 0.982 | - | 409.562 | 3686.250 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 1gib | 5120.00 +- 0.00 | 0.800 +- 0.000 | 0.800 | 3.09 | 311.23 | 123.3 | 155.0 | 573.0 | 0.893 | - | 102.375 | 920.000 | - | - | - | - |
| fixed_dual_mr09 | mamba2_1b3 | 4gib | 5120.00 +- 0.00 | 0.200 +- 0.000 | 0.200 | 2.39 | 900.01 | 12.3 | 155.0 | 109.8 | 0.982 | - | 409.562 | 3686.000 | - | - | - | - |
| avmp_static_mr05 | jamba_1_5_mini | 1gib | 5120.00 +- 0.00 | 0.600 +- 0.000 | 0.600 | 4.65 | 70.65 | 4.7 | 155.0 | 691.9 | 0.994 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.33 | 41.89 | 0.0 | 155.0 | 459.4 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.00 | 75.59 | 4.7 | 155.0 | 647.3 | 0.994 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_static_mr05 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 4.64 | 64.97 | 0.0 | 155.0 | 620.5 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 5.86 | 69.85 | 4.7 | 155.0 | 630.5 | 0.994 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | jamba_1_5_mini | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 7.74 | 55.01 | 0.0 | 155.0 | 403.8 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 1gib | 9216.00 +- 0.00 | 0.778 +- 0.000 | 0.778 | 6.13 | 68.98 | 4.7 | 155.0 | 672.7 | 0.994 | - | 512.000 | 512.000 | 0 | 0.00 | 0 | 0.00 |
| avmp_dynamic_b128 | mamba2_1b3 | 4gib | 9216.00 +- 0.00 | 0.111 +- 0.000 | 0.111 | 5.95 | 64.18 | 0.0 | 155.0 | 610.3 | 1.000 | - | 2048.000 | 2048.000 | 0 | 0.00 | 0 | 0.00 |

## Relative improvement vs padded_unified

### avmp_dynamic_b128

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| sharegpt_replay | jamba_1_5_mini | 1gib | -inf | -800.00% | +82.72% |
| sharegpt_replay | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +100.00% |
| sharegpt_replay | mamba2_1b3 | 1gib | +2.78% | -80.00% | +97.89% |
| sharegpt_replay | mamba2_1b3 | 4gib | +44.44% | -80.00% | +100.00% |

### avmp_static_mr05

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| sharegpt_replay | jamba_1_5_mini | 1gib | -inf | -400.00% | +82.72% |
| sharegpt_replay | jamba_1_5_mini | 4gib | +44.44% | -80.00% | +100.00% |
| sharegpt_replay | mamba2_1b3 | 1gib | +2.78% | -80.00% | +97.89% |
| sharegpt_replay | mamba2_1b3 | 4gib | +44.44% | -80.00% | +100.00% |

### fixed_dual_mr05

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| sharegpt_replay | jamba_1_5_mini | 1gib | -inf | -400.00% | +82.72% |
| sharegpt_replay | jamba_1_5_mini | 4gib | +0.00% | +0.00% | +100.00% |
| sharegpt_replay | mamba2_1b3 | 1gib | +0.00% | +0.00% | +97.89% |
| sharegpt_replay | mamba2_1b3 | 4gib | +0.00% | +0.00% | +100.00% |

### fixed_dual_mr09

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| sharegpt_replay | jamba_1_5_mini | 1gib | -inf | -400.00% | -356.79% |
| sharegpt_replay | jamba_1_5_mini | 4gib | +0.00% | +0.00% | -825.00% |
| sharegpt_replay | mamba2_1b3 | 1gib | +0.00% | +0.00% | +44.11% |
| sharegpt_replay | mamba2_1b3 | 4gib | +0.00% | +0.00% | +80.93% |

---

Generated: 2026-05-20 from git SHA 82af7d48f780, hardware: cuda (linux x86_64).

Regenerate: `python -m cachepawl.benchmarks.compare --quick --device cpu --output benchmarks/results/baseline/quick/`
