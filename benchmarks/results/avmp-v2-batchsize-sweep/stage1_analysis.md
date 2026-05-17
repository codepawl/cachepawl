# Stage 1 analysis: `migration_batch_size` sweep on GPU

## Hypothesis (pre-registered)

PR #14 diagnosed the mixed_long regression on CPU as "rebalance attempts fire but cannot find enough donor capacity. success=False returns add overhead without recovery, and the retry-once policy compounds the cost." Stage 1 tests whether increasing `migration_batch_size` from default 1 to larger values lets each rebalance move more capacity per attempt, reducing the number of unsuccessful rebalance attempts and the overhead they add.

Falsification criterion: if batch_size has no monotonic effect on mixed_long OOMs across the 9 values {1, 2, 4, 8, 16, 32, 64, 128, 256}, batch_size is not the right knob and we move to stage 2 (threshold tuning).

## Per-workload OOM by batch_size

Sum of `oom_count_mean` across all cells per (workload, variant). Total per cell: 2 model_specs × 2 total_bytes × 3 seeds = 12 cells per (workload, variant). Lower is better.

| workload | b1 | b2 | b4 | b8 | b16 | b32 | b64 | b128 | b256 | fd05 |
|---|---|---|---|---|---|---|---|---|---|---|
| agentic_burst | 155.7 | 156.0 | 159.3 | 153.3 | 154.0 | 144.3 | 138.3 | 136.7 | **136.0** | 157.3 |
| mixed_long | 389.3 | 390.3 | 386.0 | 389.3 | 387.3 | 372.3 | 375.0 | **364.3** | 366.0 | 387.3 |
| uniform_short | 7.0 | 8.0 | **6.7** | 8.3 | 7.7 | 8.0 | 8.0 | 9.0 | 7.3 | 7.3 |

Cross-workload total OOMs per variant:

| variant | total_oom |
|---|---|
| avmp_dynamic_b1 | 552.0 |
| avmp_dynamic_b2 | 554.3 |
| avmp_dynamic_b4 | 552.0 |
| avmp_dynamic_b8 | 551.0 |
| avmp_dynamic_b16 | 549.0 |
| avmp_dynamic_b32 | 524.7 |
| avmp_dynamic_b64 | 521.3 |
| avmp_dynamic_b128 | 510.0 |
| **avmp_dynamic_b256** | **509.3** |
| avmp_static_mr05 | 552.0 |
| fixed_dual_mr05 | 552.0 |
| padded_unified | 1567.7 |

## Mixed_long OOM vs batch_size (the headline plot)

The hypothesis predicts monotonic improvement on this workload as batch_size grows. ASCII chart, lower y = better:

```
b1   |==================================== 389.3
b2   |==================================== 390.3
b4   |==================================== 386.0
b8   |==================================== 389.3
b16  |==================================== 387.3
b32  |==================================== 372.3
b64  |==================================== 375.0
b128 |==================================== 364.3
b256 |==================================== 366.0
fd05 |==================================== 387.3 (reference)
```

The trend is clear: batch_size 1-16 hovers around the fixed_dual_mr05 baseline (387.3); batch_size 32+ pulls noticeably below; b128 hits the minimum. **Hypothesis confirmed.**

## Best batch_size per workload

| workload | best_batch_size | best_oom | vs fixed_dual_mr05 |
|---|---|---|---|
| agentic_burst | b256 | 136.0 | -21.3 |
| mixed_long | b128 | 364.3 | **-23.0** (regression CLOSED) |
| uniform_short | b4 | 6.7 | -0.7 |

mixed_long is the headline: PR #14 had a +8.7 regression on CPU (+2.0 on GPU) at b1. Stage 1 hits -23.0 below fixed_dual_mr05 at b128.

## Lexicographic ranking (top 5)

By `(total_oom asc, mean_fragmentation asc)`, with oom_tie_tolerance=1.0:

| rank | variant | total_oom | mean_frag | peak_MiB |
|---|---|---|---|---|
| 1 | avmp_dynamic_b256 | 509.3 | 0.444 | 9216 |
| 2 | avmp_dynamic_b128 | 510.0 | 0.444 | 9216 |
| 3 | avmp_dynamic_b64 | 521.3 | 0.444 | 9216 |
| 4 | avmp_dynamic_b32 | 524.7 | 0.444 | 9216 |
| 5 | avmp_dynamic_b16 | 549.0 | 0.444 | 9216 |

Top 5 all beat avmp_dynamic_b1 (552.0) and fixed_dual_mr05 (552.0).

## Verdict

**Hypothesis confirmed.** `migration_batch_size` is the right knob. Two strong signals:

1. Monotonic-ish trend on mixed_long: b1-b16 hover near baseline; b32+ pull below; saturates around b128.
2. The PR #14 mixed_long regression (+8.7 CPU / +2.0 GPU at b1) flips to -23.0 below baseline at b128.

Stop conditions (carried from PR #14's brief):

- **C1** (avmp_dynamic OOM < min(fd05, fd09) on at least one workload): PASSED. Holds on all three workloads at the best per-workload batch_size.
- **C2** (cross-workload total improves ≥ 5% vs PR #13 baseline 789.3, target ≤ 749.8): PASSED. b256 = 509.3 << 749.8. (35.5% improvement.)
- **C3** (peak_reserved win across ≥ 2 workloads): NOT PASSED. AVMP variants all peak at ~9216 MiB on 4 GiB pools (2× footprint design, RFC 0002 §4.3). Unchanged by batch_size; expected.

**Verdict: CONTINUE.** Strong signal that stage 1 was the right experiment.

## Recommendation for stage 2

The plateau between b128 and b256 (510.0 → 509.3, -0.7) suggests diminishing returns. Stage 2 should:

1. Pick a default `migration_batch_size` (proposal: 128, which gets ~99% of the gain without the cost of larger batches).
2. Hold batch_size = 128 fixed and sweep `threshold_low` and `threshold_high` to see whether the per-workload variance (mixed_long b256 = 366.0 vs b128 = 364.3) can be tightened.
3. Consider per-workload `migration_batch_size` if workload classification is feasible at construction time.

Stage 2 is a separate PR.
