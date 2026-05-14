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
| padded_unified | jamba_1_5_mini | 1gib | 0.00 +- 0.00 | 0.682 +- 0.000 | 0.995 | 2.09 | 103.88 | 0.0 | 0.000 | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.202 +- 0.000 | 0.961 | 2.33 | 1460.41 | 5.0 | - | 512.000 | 512.000 |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.219 +- 0.000 | 0.969 | 2.26 | 712.46 | 247.0 | - | 102.375 | 921.500 |

## Relative improvement vs padded_unified

### fixed_dual_mr05

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| uniform_short | jamba_1_5_mini | 1gib | +70.33% | -150.00% | -inf |

### fixed_dual_mr09

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| uniform_short | jamba_1_5_mini | 1gib | +67.83% | -29.98% | -inf |

---

Generated: 2026-05-14 from git SHA 37cb9ff4e48a, hardware: cpu (linux x86_64).

Regenerate: `python -m cachepawl.benchmarks.compare --quick --device cpu --output benchmarks/results/baseline/quick/`
