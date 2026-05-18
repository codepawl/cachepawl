# Tier 1 PR B throughput analysis

Target variant: `avmp_dynamic_b128`. Baseline for ratio comparison: `fixed_dual_mr05`.

## Pre-registered stop rule

The throughput claim is justified iff EITHER `effective_batch_size_p50` >= 1.05x baseline on at least 2 workloads OR `goodput_requests_per_second` >= 1.10x baseline on at least 1 workload. The criterion was registered in the design doc before the sweep was run.

**Verdict: PASS_GOODPUT**

- Workloads where eff_batch_size_p50 ratio >= 1.05: (none) (threshold: 2)
- Workloads where goodput ratio >= 1.10: agentic_burst, mixed_long, uniform_short (threshold: 1)

## Hypothesis evaluation (per-workload ratios)

| workload | eff_batch_p50 (target) | eff_batch_p50 (baseline) | ratio | goodput (target) | goodput (baseline) | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| agentic_burst | 129.00 | 129.00 | 1.000x | 46.91 | 25.69 | 1.826x |
| mixed_long | 132.00 | 132.00 | 1.000x | 65.07 | 27.25 | 2.388x |
| uniform_short | 284.00 | 284.00 | 1.000x | 434.24 | 32.65 | 13.299x |

## Cross-workload lexicographic ranking (3-level)

Sort key: `(total_oom asc, effective_batch_size_p50 desc, fragmentation_during_load asc)`. Lower OOM wins; within tie-tolerance, higher sustained batch size wins; remaining ties break on lower fragmentation.

```
rank  variant_label              total_oom  eff_batch_p50  mean_frag   peak_MiB
----  ------------------------  ----------  -------------  ---------  ---------
   1  avmp_dynamic_b128              510.0         181.67      0.444       9216
   2  avmp_static_mr05               552.0         181.67      0.430       8875
   3  fixed_dual_mr05                552.0         181.67      0.500       5120
   4  fixed_dual_mr09               1221.3         181.67      0.500       5120
   5  padded_unified                1567.7         181.67      0.433       4779
```

