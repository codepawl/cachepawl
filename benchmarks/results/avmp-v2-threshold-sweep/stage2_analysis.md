# Stage 2 analysis: threshold tuning at fixed `migration_batch_size=128`

## Hypotheses (pre-registered)

- **Hypothesis A** (threshold_high): lowering `threshold_high` from 0.30 makes the donor pool qualified more often, helping `mixed_long` where both pools sit between 0.05 and 0.30. Tested at th_high in {0.10, 0.20}; default 0.30 is the stage 1 reference.
- **Hypothesis B** (threshold_low): lowering `threshold_low` from 0.05 fires the trigger earlier, potentially preventing OOMs on `uniform_short` and `agentic_burst`, at the risk of thrashing. Tested at th_low in {0.02, 0.10}; default 0.05 is the stage 1 reference.

Pre-registered stop rule: stage 2 best cross-workload `total_oom` must be **≤ 494.0** (3% improvement vs stage 1 `avmp_dynamic_b128` baseline of 510.0) to declare threshold tuning materially better. Otherwise, declare `migration_batch_size` the dominant axis and recommend stage 1 `b128` for the paper config.

## Sweep shape

7 variants (3 baselines + 4 threshold variants) x 3 workloads x 2 model_specs x 2 total_bytes options (1 GiB, 4 GiB after `--max-total-bytes 4294967296`) x 3 seeds = 252 cells. GPU: RTX 3060, CUDA 13, torch 2.12.0+cu130. Wall: 19m47s.

The stage 1 `avmp_dynamic_b128` row (threshold_low=0.05, threshold_high=0.30) is the reference point and is NOT re-run; its data is pulled from `benchmarks/results/avmp-v2-batchsize-sweep/aggregated.json` and inlined below.

## Hypothesis A: threshold_high (lower triggers donor qualification earlier on mixed_long)

`mixed_long` is the workload where stage 1 saw the binding constraint at b1 (PR #14's regression). Sum of `oom_count_mean` across the 12 mixed_long cells per variant. Lower is better.

| threshold_high | label                         | mixed_long_oom | agentic_burst_oom | uniform_short_oom | total_oom |
|---:|---|---:|---:|---:|---:|
| 0.10 | `avmp_dynamic_b128_th_high_010` | 364.33 | 136.67 | 9.00 | 510.00 |
| 0.20 | `avmp_dynamic_b128_th_high_020` | 364.33 | 136.67 | 9.00 | 510.00 |
| 0.30 (default, stage 1 ref) | `avmp_dynamic_b128`            | 364.33 | 136.67 | 9.00 | 510.00 |

**Hypothesis A: NOT supported.** Lowering `threshold_high` from 0.30 to 0.20 to 0.10 produces zero change on `mixed_long`, on the other two workloads, and on the total. The rebalancer's trigger does not lie in this band on this workload mix.

## Hypothesis B: threshold_low (lower fires the trigger earlier on uniform_short / agentic_burst)

Sum of `oom_count_mean` for the two workloads pre-registered for this hypothesis, plus the sum of `allocator_specific_median.rebalance_count` across the variant's 12 cells. The trigger-rate proxy is `rebalance_count`; an earlier-firing trigger is expected to push it up.

| threshold_low | label                         | uniform_short_oom | agentic_burst_oom | rebalance_count_sum | total_oom |
|---:|---|---:|---:|---:|---:|
| 0.02 | `avmp_dynamic_b128_th_low_002`  | 9.00 | 136.67 | 84 | 510.00 |
| 0.05 (default, stage 1 ref) | `avmp_dynamic_b128`            | 9.00 | 136.67 | 84 | 510.00 |
| 0.10 | `avmp_dynamic_b128_th_low_010`  | 9.00 | 136.67 | 84 | 510.00 |

**Hypothesis B: NOT supported.** `threshold_low` variation in [0.02, 0.10] produces no change on either pre-registered workload, no change in `rebalance_count`, no change in `bytes_migrated_total` (3584 MiB across all four variants), and no thrashing (`auto_rebalance_skipped_throttle = 0` everywhere).

## Why the thresholds are inert here

Cross-checking the avmp_dynamic stats across all four threshold variants:

| variant | rebalance_count_sum | MiB_migrated_sum | time_rebalancing_ms_sum | cross_pool_eviction | throttle_skips |
|---|---:|---:|---:|---:|---:|
| `avmp_dynamic_b128_th_high_010` | 84 | 3584.00 | 37.17 | 0 | 0 |
| `avmp_dynamic_b128_th_high_020` | 84 | 3584.00 | 39.92 | 0 | 0 |
| `avmp_dynamic_b128_th_low_002`  | 84 | 3584.00 | 39.01 | 0 | 0 |
| `avmp_dynamic_b128_th_low_010`  | 84 | 3584.00 | 36.73 | 0 | 0 |

Only `time_rebalancing_ms_sum` varies (wall-clock noise). Counters are bit-identical across all four configurations, on all three workloads. The rebalancer fires the same number of times and migrates the same amount of capacity regardless of where the threshold pair lies in [0.02, 0.30].

Mechanism (inferred from pool inspection): the pool occupancies at the moments the auto-rebalance trigger evaluates are either far below `threshold_low` (so the donor is not qualified) or already saturated at the recipient (so the trigger fires regardless of where `threshold_high` sits). The interval [0.02, 0.30] does not span the actual occupancies observed during these workloads.

## Lexicographic ranking (combined stage 1 b128 + stage 2)

By `(total_oom asc, mean_fragmentation asc)`, with oom_tie_tolerance = 1.0.

| rank | variant | total_oom | mean_frag | peak_MiB |
|---|---|---:|---:|---:|
| 1 (tie) | `avmp_dynamic_b128_th_high_010` | 510.00 | 0.4444 | 9216 |
| 1 (tie) | `avmp_dynamic_b128_th_high_020` | 510.00 | 0.4444 | 9216 |
| 1 (tie) | `avmp_dynamic_b128_th_low_002`  | 510.00 | 0.4444 | 9216 |
| 1 (tie) | `avmp_dynamic_b128_th_low_010`  | 510.00 | 0.4444 | 9216 |
| 1 (tie) | `avmp_dynamic_b128` (stage 1)   | 510.00 | 0.4444 | 9216 |
| 6 | `avmp_static_mr05` | 552.00 | 0.4296 | 9216 |
| 7 | `fixed_dual_mr05`  | 552.00 | 0.5000 | 5120 |
| 8 | `padded_unified`   | 1567.67 | 0.4333 | 5120 |

All five top-ranked variants are byte-identical at the counter level on every workload. The rank order between them is incidental (alphabetical on label).

## Interaction analysis caveat

This sweep is orthogonal, not factorial: each variant changes one threshold and holds the other at its default. We cannot rule out an interaction effect (e.g., simultaneously lowering both thresholds, or pairing a low threshold with a smaller `migration_batch_size`). Given the strength of the null result here (identical counters), the simplest reading is that interaction effects are also inert in this neighborhood, but a 2D sweep would be needed to confirm.

## Pre-registered stop rule check

- Pre-registered target: stage 2 best `total_oom` <= 494.0 (3% improvement vs stage 1 b128 = 510.0).
- Observed: stage 2 best `total_oom` = **510.00**.
- 510.00 > 494.00. **Stop rule NOT met.**

**Verdict: marginal.** Threshold tuning in [0.02, 0.30] is not materially better than the stage 1 default. `migration_batch_size` remains the dominant axis identified by these two stages.

## Paper config recommendation

Use **stage 1's `avmp_dynamic_b128`** (threshold_low=0.05, threshold_high=0.30, migration_batch_size=128) for the paper. The stage 2 variants tie exactly with this point; choosing the simpler default avoids implying that the threshold value was selected on the basis of evidence it doesn't have.

## Determinism check

Reran `avmp_dynamic_b128_th_high_010` (first variant in the alphabetical tie) into a fresh output directory and diffed against the original 36 cells. Logical counters compared:

- `metrics.oom_count`, `metrics.preemption_count`, `metrics.active_requests_samples`
- `metrics.allocator_specific_stats.{rebalance_count, bytes_migrated_total, cross_pool_eviction_count, auto_rebalance_skipped_throttle, bytes_wasted_to_alignment_total, virtual_handles_live, *_used, *_free, *_total, *_ratio, current_*}`

Result: 36/36 cells byte-identical on the deterministic key set. `fragmentation_samples` (derived from `torch.cuda.memory_reserved`) drifts across reruns as expected from CUDA caching; this matches the protocol established in PR #15.

## What this sweep does NOT establish

- 2D interactions between `threshold_low` and `threshold_high`.
- Behavior outside [0.02, 0.30].
- Effects with `migration_batch_size != 128`.
- `min_rebalance_interval_ops` tuning.
- Behavior on workloads where pool occupancies actually traverse the [0.02, 0.30] band (the three tested workloads do not).
