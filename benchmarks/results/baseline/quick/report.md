# Allocator baseline comparison

## How to read

- `peak_reserved_MiB`: max bytes the pool held during the run (lower is better).
- `fragmentation_ratio`: final `1 - allocated/reserved` sample (lower is better).
- `alloc_p50_us` / `alloc_p99_us`: allocate-call latency in microseconds (latency varies across reruns).
- `oom_count`: number of `OutOfMemoryError` raised during the run (lower is better).
- `padding_waste_MiB`: bytes wasted by padded_unified rounding SSM blocks up to the KV page size. Only meaningful for `padded_unified` rows.
- `kv_underused_MiB`, `ssm_underused_MiB`: bytes in each fixed_dual pool that were reserved but unused. Only meaningful for `fixed_dual_*` rows.
- `mean +- std`: mean across replicates with population standard deviation (ddof=0).

## Workload: uniform_short

| variant | model_spec | total_bytes | peak_reserved_MiB | fragmentation_ratio | alloc_p50_us | alloc_p99_us | oom_count | padding_waste_MiB | kv_underused_MiB | ssm_underused_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_unified | jamba_1_5_mini | 1gib | 0.00 +- 0.00 | 1.000 +- 0.000 | 2.04 | 98.55 | 3.0 | 33.188 | - | - |
| fixed_dual_mr05 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.801 +- 0.000 | 2.91 | 6621.65 | 13.0 | - | 1431.000 | 2344.000 |
| fixed_dual_mr09 | jamba_1_5_mini | 1gib | 0.01 +- 0.00 | 0.694 +- 0.000 | 3.12 | 2295.74 | 243.0 | - | 65016.000 | 1550.188 |

## Relative improvement vs padded_unified

### fixed_dual_mr05

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| uniform_short | jamba_1_5_mini | 1gib | +19.88% | -150.00% | -333.33% |

### fixed_dual_mr09

| workload | model_spec | total_bytes | fragmentation_pct_better | peak_reserved_pct_better | oom_pct_better |
| --- | --- | --- | --- | --- | --- |
| uniform_short | jamba_1_5_mini | 1gib | +30.60% | -29.98% | -8000.00% |

---

Generated: 2026-05-14 from git SHA 557992298f92, hardware: cpu (linux x86_64).

Regenerate: `python -m cachepawl.benchmarks.compare --quick --device cpu --output benchmarks/results/baseline/quick/`
