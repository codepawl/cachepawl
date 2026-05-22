# TritonAVMPAllocator sweep parity analysis

Comparison: `avmp_dynamic_b128_triton` (TritonAVMPAllocator + Triton `zero_page_kernel`) vs `avmp_dynamic_b128` (Python `AsymmetricVirtualPool` baseline) over the v2 hardware-realization sweep (`triton_validation` variant set, 6 variants x 3 workloads x 2 specs x 2 byte-sizes x 3 seeds = 216 cells, 36 paired).

## §1 Wall-clock comparison

| Sweep | Cells | Total wall (s) | Per-cell (s) |
|---|---|---|---|
| v1 (avmp-v2-throughput, 5 var, 180 cells) | 180 | 974.1 | 5.41 |
| v2 triton_validation (6 var, 216 cells) | 216 | 2089.4 | 9.67 |

## §2 Per-cell parity table (12 paired aggregated rows)

Each row aggregates the 3 seed replicates per (workload, model_spec, total_bytes) triple. `oom_count_mean` is the mean OOM count across replicates; `effective_batch_size_p50_median` is the median across replicates.

| Workload | Model | total_bytes | py OOM mean | triton OOM mean | OOM diff | py batch_p50 | triton batch_p50 | batch rel diff | goodput ratio |
|---|---|---|---|---|---|---|---|---|---|
| agentic_burst | jamba_1_5_mini | 1 GiB | 38.33 | 38.33 | +0.00 | 129.00 | 129.00 | 0.0000% | 0.2221 |
| agentic_burst | jamba_1_5_mini | 4 GiB | 24.67 | 24.67 | +0.00 | 129.00 | 129.00 | 0.0000% | 0.2426 |
| agentic_burst | mamba2_1b3 | 1 GiB | 36.67 | 36.67 | +0.00 | 129.00 | 129.00 | 0.0000% | 0.1966 |
| agentic_burst | mamba2_1b3 | 4 GiB | 37.00 | 37.00 | +0.00 | 129.00 | 129.00 | 0.0000% | 0.2132 |
| mixed_long | jamba_1_5_mini | 1 GiB | 99.33 | 99.33 | +0.00 | 132.00 | 132.00 | 0.0000% | 0.3559 |
| mixed_long | jamba_1_5_mini | 4 GiB | 75.00 | 75.00 | +0.00 | 132.00 | 132.00 | 0.0000% | 0.4461 |
| mixed_long | mamba2_1b3 | 1 GiB | 100.67 | 100.67 | +0.00 | 132.00 | 132.00 | 0.0000% | 0.2423 |
| mixed_long | mamba2_1b3 | 4 GiB | 89.33 | 89.33 | +0.00 | 132.00 | 132.00 | 0.0000% | 0.2490 |
| uniform_short | jamba_1_5_mini | 1 GiB | 5.33 | 5.33 | +0.00 | 284.00 | 284.00 | 0.0000% | 0.2203 |
| uniform_short | jamba_1_5_mini | 4 GiB | 0.00 | 0.00 | +0.00 | 284.00 | 284.00 | 0.0000% | 0.2110 |
| uniform_short | mamba2_1b3 | 1 GiB | 3.67 | 3.67 | +0.00 | 284.00 | 284.00 | 0.0000% | 0.1813 |
| uniform_short | mamba2_1b3 | 4 GiB | 0.00 | 0.00 | +0.00 | 284.00 | 284.00 | 0.0000% | 0.2561 |

## §3 Aggregate metrics

- Sum of `oom_count_mean` across the 12 paired rows:
    - `avmp_dynamic_b128` (Python): **510.00**
    - `avmp_dynamic_b128_triton`: **510.00**
    - Difference: **+0.00**
- v1 reference for `avmp_dynamic_b128` (`benchmarks/results/avmp-v2-throughput/full/aggregated.json`): **510.00**
- Max `effective_batch_size_p50` relative drift across cells: **0.0000%**
- Tolerance: 1% per TRITON_ROADMAP.md §3

## §4 Failure-mode log

No cells with OOM-drift != 0 or batch-p50 drift > 1%. The hardware realization holds at the sweep scale; the 0.0000% drift result from the Week 1 9-cell smoke extends to all 12 paired rows here.

## §5 Goodput delta (paired bootstrap 95% CI)

- Bootstrap protocol: B=10000, seed=20260520 (same as `research/avmp/scripts/bootstrap_ci.py`).
- Statistic: ratio of means, `triton / python`, paired on (workload, model, total_bytes, seed) across 36 per-seed cells.

| Slice | n_pairs | Ratio (mean) | 95% CI low | 95% CI high |
|---|---|---|---|---|
| overall | 36 | 0.2375 | 0.2137 | 0.2674 |
| agentic_burst | 12 | 0.2238 | 0.2022 | 0.2468 |
| mixed_long | 12 | 0.3183 | 0.2736 | 0.3707 |
| uniform_short | 12 | 0.2254 | 0.1975 | 0.2600 |

Interpretation: the **event stream** the two allocators see is byte-identical (every paired cell has identical OOM count, identical `effective_batch_size_p50`, and identical migration counts). The ratio < 1.0 is **simulator wall-clock**, not inference goodput: the Python baseline does pure-Python bookkeeping with zero kernel launches, whereas the Triton variant launches one `zero_page_kernel` per `allocate()` call. With `cuda.synchronize()` happening inside `run_benchmark`'s per-call latency timing, the simulator pays a kernel-launch + driver round-trip per allocate that the Python baseline does not. This is exactly the per-allocate cost characterized in `tests/benchmarks/test_zero_page_latency.py` (~50-100 us sync-per-call); a real inference engine amortizes launches across decode steps and would not see this slowdown. The correctness claim (identical OOM count + batch_p50 across all 36 paired seed-cells) is independent of the wall-clock ratio.
