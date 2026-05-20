# AVMP paper numerical claim audit

**Date**: 2026-05-20
**Branch**: `audit/numerical-claims` (off `feat/paper-v15-phase4` tip `4f64ce3`)
**Scope**: every numerical claim in `research/avmp/sections/*.tex` cross-checked against committed source data.

## Summary

**73 claims audited:**
- **MATCH: 60**
- **MISMATCH: 5** (five distinct issues; one issue replicated across 4+ sections counts as one MISMATCH here, with all paper locations listed)
- **UNTRACED: 1**
- **EXTERNAL_FLAG: 4**
- **NOTE (acceptable rounding / multi-statistic): 3**

Reproducibility smoke: `scripts/bootstrap_ci.py` re-run produces byte-identical `bootstrap_ci.json` (md5 unchanged). Throughput sweep grid verified directly from `aggregated.json` row enumeration.

**Verdict before arXiv submission: NOT READY.** Five MISMATCHes need decisions before fixing. The biggest is the sweep-grid misdescription that propagates across the abstract, §1, §4, and conclusion.

---

## MISMATCHes (need Phase 2 fixes — see decision points)

### M1. Sweep grid: 270 vs 180 cells, 3 vs 2 pool sizes, zamba2_2_7b vs mamba2_1b3

**Paper claim**: "270 cells: 5 allocator variants, 3 synthetic workloads, 2 model specifications (`jamba_1_5_mini` and `zamba2_2_7b`), 3 total memory pool sizes (1 GiB, 4 GiB, 8 GiB), and 3 random seeds."

**Source truth** (`benchmarks/results/avmp-v2-throughput/full/aggregated.json` direct row enumeration):
- variants: 5 (`avmp_dynamic_b128`, `avmp_static_mr05`, `fixed_dual_mr05`, `fixed_dual_mr09`, `padded_unified`) ✓
- workloads: 3 (`agentic_burst`, `mixed_long`, `uniform_short`) ✓
- **models**: `jamba_1_5_mini` and **`mamba2_1b3`** (NOT `zamba2_2_7b`)
- **pool sizes**: only `1 GiB` and `4 GiB` (NO `8 GiB`)
- seeds: 3 ✓
- Resulting grid: 5 × 3 × 2 × **2** × 3 = **180 cells**, not 270.
- `aggregated.json` has 60 rows (one per cell, averaging 3 seeds) = 5 × 3 × 2 × 2.

**Paper locations with the wrong "270":**
- `sections/00-abstract.tex:11` — "270 synthetic cells plus 60 cells of ShareGPT trace replay"
- `sections/01-introduction.tex:25` — "270-cell sweep evaluating 5 allocator variants, 3 synthetic workloads, 3 pool sizes"
- `sections/05-evaluation.tex:7` — "an experimental sweep across 270 cells: 5 allocator variants, 3 synthetic workloads, 2 model specifications (`jamba_1_5_mini` and `zamba2_2_7b`), 3 total memory pool sizes (1 GiB, 4 GiB, 8 GiB), and 3 random seeds"
- `sections/05-evaluation.tex:171` — "synthetic prompt distributions... 270-cell sweep"

**Decision needed**: either
- (a) update the paper to "180 cells, 2 pool sizes (1/4 GiB), models jamba_1_5_mini + mamba2_1b3" everywhere, or
- (b) re-run the sweep with 8 GiB pool and `zamba2_2_7b` to actually have 270 cells.
- (a) is the cheaper honest fix; (b) is a real engineering re-sweep that would change other numbers downstream.

### M2. Time-decomposition sweep: 90 cells claim

**Paper claim** (`sections/05-evaluation.tex:7`): "90 cells, 18:46"

**Source truth** (`benchmarks/results/avmp-v15-timedecomp/aggregated.json`): 60 aggregated rows (5 × 3 × 2 × 2 = 60); raw cell count 180 (× 3 seeds). Wall time of 18:46 matches `SWEEP_METADATA.json` `total_wall_seconds=1126` ✓.

**Decision**: cell count should be "180 cells" or "60 cells (180 raw including replicates)". The 90 figure is wrong by exactly the same factor as M1 (off by × 1.5, consistent with a missing dimension).

### M3. Migration churn absolute values: 313 / 352 MiB

**Paper claim** (`sections/05-evaluation.tex:111`): "b128 migrates 313 MiB versus b256's 352 MiB per cell on average"

**Source truth** (`benchmarks/results/avmp-v2-batchsize-sweep/aggregated.json`, `allocator_specific_median.bytes_migrated_total`, averaged across 12 cells per variant):
- b128: **298.67 MiB**
- b256: **336.00 MiB**

Paper numbers are +14.33 / +16.00 MiB higher than committed data; differ by ~4.8% systematically.

**Decision**: source of the 313/352 numbers cannot be located in current committed data. May be stale (pre-replicate-bump?) or from a different aggregation (sum of per-workload averages vs grand mean). Need to either reconcile or update prose. The 12.5% reduction claim in M4 is mathematically inconsistent with both versions, suggesting the absolute numbers are stale and were never recomputed.

### M4. Migration churn reduction percentage: 12.5%

**Paper claim** (`sections/05-evaluation.tex:111`): "12.5% reduction in migration churn"

**Source truth** (paper's own arithmetic, plus data check):
- (352-313)/352 = 11.08% (paper's own numbers don't yield 12.5%)
- (336-298.67)/336 = 11.11% (committed data)

Neither computation yields 12.5%. Either:
- The 313/352 numbers AND the 12.5% are both stale
- Or the 12.5% is from a different metric (peak vs total, or per-workload vs mean)

**Decision**: most likely both M3 and M4 need to be replaced with the true values (~298.67 MiB vs 336.00 MiB, 11.1% reduction). Confirm by re-running `scripts/generate_tables.py` and re-citing.

### M5. Goodput-ratio prose vs bootstrap point-estimate inconsistency

**Paper claim** (`sections/05-evaluation.tex:55`, abstract, intro, conclusion, etc.):
- `uniform_short`: "434.24 vs 32.65 req/s, **13.30×**" with CI `[11.18, 16.00]`
- `mixed_long`: "65.07 vs 27.25 req/s, **2.39×**" with CI `[1.70, 3.04]`
- `agentic_burst`: "46.91 vs 25.69 req/s, **1.83×**" with CI `[1.42, 2.60]`

**Source truth**:
- `table_per_workload_winner.tex`: 13.30 / 2.39 / 1.83 (mean of cell medians; cells = (model, pool), 4 per workload)
- `bootstrap_ci.json`: 12.93 / 2.19 / 1.83 (mean-of-cell-means via ratio-of-paired-means resampling on the 12-cell (model, pool, seed) grid)

The CI bounds quoted in prose (`[11.18, 16.00]`, etc.) match `bootstrap_ci.json` exactly. But the point estimate quoted in prose (13.30) matches `table_per_workload_winner.tex` instead, not the bootstrap point. So the paper effectively says "the point estimate is 13.30 with 95% CI [11.18, 16.00]" — but the CI was bootstrapped around a point of 12.93, not 13.30. The CI happens to cover 13.30, so the inference holds, but the presentation is misleading.

**Decision needed (subtle)**:
- (a) Acknowledge the two-statistic gap with a footnote: "ratio of cell medians; bootstrap CI centered around the per-seed-cell mean ratio of 12.93."
- (b) Change the prose point estimate to 12.93 / 2.19 / 1.83 (matches bootstrap), regenerate `table_per_workload_winner.tex` accordingly. This is the cleaner statistical story.
- (c) Bootstrap the ratio-of-cell-medians instead of ratio-of-paired-means, so point and CI use the same statistic. This requires a bootstrap script change.

(b) is the easiest. (c) preserves the higher 13.30 number but adds engineering work.

---

## UNTRACED

### U1. Table 4 (`table_sharegpt_results.tex`) bootstrap CIs

The CIs in `table_sharegpt_results.tex` (e.g., `2.36× [1.33, 5.65]`) are NOT in `bootstrap_ci.json` (which only holds the synthetic 270-cell/180-cell sweep results). They are produced by `analyze_sharegpt.py` separately. The audit did NOT re-run `analyze_sharegpt.py` to confirm byte-identical reproduction; verbal cross-check of the rendered values vs prose found them internally consistent. To upgrade to MATCH, re-run the script and md5-check the output against the committed table.

---

## EXTERNAL_FLAGs (cannot verify from local data; manual review before arXiv)

### E1. vLLM issue #37121 numbers: 7.3× capacity overestimation, 13.7% effective VRAM utilization

Appears in:
- `sections/00-abstract.tex:11` — "wasting up to $7.3\times$ capacity"
- `sections/03-method.tex:20` — "7.3$\times$ KV cache overestimation on Qwen3.5-4B-AWQ, leaving 13.7\% effective VRAM utilization at peak memory" (the canonical detailed statement, retained after the §1.1/§2.3 dedupe pass)
- `sections/01-introduction.tex:9`, `sections/02-background.tex:36` — short citations only

**Manual check before arXiv**: verify the 7.3× and 13.7% match the current state of vLLM issue #37121.

### E2. Jamba 1.5 Mini H100 deployment numbers (§6.3)

`sections/07-discussion.tex:17`: "A Jamba 1.5 Mini deployment requires approximately 24 GiB VRAM per H100 80GB instance for model weights and activations, leaving approximately 56 GiB for KV cache and SSM state combined."

**Manual check**: confirm 24 GiB / 56 GiB figures against an authoritative Jamba 1.5 Mini deployment guide.

### E3. SGLang `mamba_full_memory_ratio` default of 0.9

Used as a load-bearing fact across `sections/01-introduction.tex:11`, `02-background.tex:38`, `03-method.tex:165`, `06-related-work.tex:12`, `07-discussion.tex:17`. All instances consistently say 0.9.

**Manual check**: confirm against the SGLang version pinned in `bibliography/refs.bib`'s `zheng2024sglang` citation.

### E4. Citation page numbers in `bibliography/refs.bib`

Includes specific page ranges for `dao2022flashattention (pp.16344-16359)`, `prabhu2024vattention (pp.1133-1150)`, `yu2022orca (pp.521-538)`, `patel2024splitwise (pp.118-132)`. Cannot be verified without the published versions.

---

## NOTES (acceptable rounding / multi-precision presentation)

### N1. `13.3` vs `13.30` split

- Abstract, Intro (§1.1, §1.2), Method (§3.2), Conclusion: use `13.3×`
- Evaluation (§4.3 prose and §4.4 ShareGPT): use `13.30×`

This is allowed presentation rounding per the audit rules. Self-consistent within each section group.

### N2. `1221` vs `1221.3` split

- Narrative sections (Intro, Background, Related Work, Discussion): `1221` (rounded)
- Method §3.2 line 165 and Evaluation §4.2 line 41: `1221.3` (full precision)

Acceptable; consistent with rounding convention.

### N3. Bootstrap CI prose rounding

- `[-1.3, -10.3]` in prose vs `[-1.33, -10.33]` in `table_bootstrap_ci.tex`
- `[-9.3, -1.3]` in prose vs `[-9.25, -1.33]` in table
- `[0.0, +1.3]` in prose vs `[0.00, 1.25]` in table

All are within 0.1 of one another and conform to standard "report-to-tenths" prose vs "report-to-hundredths" table rule. Acceptable.

---

## MATCH inventory

The following claims are fully traceable to committed source and reproduce byte-identically:

### Cross-workload OOM totals (B1-B7)

| Variant | Claim | Source value | Status |
|---|---|---|---|
| fixed_dual_mr05 cross-workload | 552 | sum(oom_count_mean over 12 cells) = 552.0 | MATCH (×9 paper locations) |
| avmp_dynamic_b128 cross-workload | 510 | sum = 510.0 | MATCH (×8 paper locations) |
| fixed_dual_mr09 cross-workload | 1221 / 1221.3 | sum = 1221.333... | MATCH (rounded) |
| padded_unified cross-workload | 1567.7 | sum = 1567.667 | MATCH |
| avmp_dynamic_b256 cross-workload | 509.3 | sum = 509.333 | MATCH |
| Per-workload Table 1 cells (15 entries) | per-cell sums | aggregated.json `oom_count_mean` per cell, summed over 4 (model, pool) cells per workload | All 15 cells MATCH |
| Figure 3 bar labels (12 bars) | individual workload OOMs | derived from same data | All MATCH |

### Bootstrap CIs (C1-C15)

All 13 CIs in `table_bootstrap_ci.tex` reproduce byte-identically from `bootstrap_ci.json` (md5 unchanged after `scripts/bootstrap_ci.py` rerun with RNG seed 20260520, B=10000). Prose CI quotations in §4.2, §4.3, §4.4 cross-check against the table within stated rounding tolerance (N3).

`n_pairs` values 12 (per-workload) and 36 (cross-workload) match the actual grid: 2 models × 2 pools × 3 seeds = 12 per workload; × 3 workloads = 36. MATCH.

### Wall-clock and resource (D1-D7)

| Claim | Paper value | Source | Status |
|---|---|---|---|
| 270-cell sweep wall time | 16:14 | `avmp-v2-throughput/full/SWEEP_METADATA.json` `total_wall_seconds=974.09` = 16:14 | MATCH (cell count separately wrong, see M1) |
| timedecomp wall time | 18:46 | `avmp-v15-timedecomp/SWEEP_METADATA.json` `total_wall_seconds=1126` = 18:46 | MATCH (cell count separately wrong, see M2) |
| ShareGPT wall time | 1:56 | `avmp-v15-sharegpt/SWEEP_METADATA.json` `total_wall_seconds=116.06` = 1:56 | MATCH |
| Service time 32.2 s vs 2.1 s on `uniform_short` 4 GiB jamba | 32.2 / 2.1 | `avmp-v15-timedecomp/aggregated.json` `time_in_service_ns_median` = 32.188 / 2.104 | MATCH |
| OOM-retry 26.3% vs 8.5% on `mixed_long` | 26.3 / 8.5 | derived `time_in_oom_retry_ns / total wall ns` summed over 12 cells per variant | MATCH |
| OOM-retry 10.0% vs 2.1% on `agentic_burst` | 10.0 / 2.1 | same derivation | MATCH |
| Peak VRAM 9216 vs 5120 MiB | 9216 / 5120 | `peak_reserved_bytes_mean` per variant on 4 GiB cells | MATCH |

### Workload + config parameters (E1-E8)

| Claim | Paper value | Source | Status |
|---|---|---|---|
| migration_batch_size sweep range | {1, 2, 4, 8, 16, 32, 64, 128, 256} | distinct `migration_batch_size` values in `avmp-v2-batchsize-sweep/aggregated.json` | MATCH |
| Threshold defaults | `threshold_high=0.30`, `threshold_low=0.05` | `table_parameter_defaults.tex` | MATCH |
| min_rebalance_interval_ops | 1000 | `table_parameter_defaults.tex` | MATCH |
| ShareGPT prompt clamp | [16, 4096] | `src/cachepawl/benchmarks/harness/workloads.py:218-228` | MATCH |
| ShareGPT generation log-normal | mean=4.5 sigma=1.0 clipped [32, 2048] | same source | MATCH |
| ShareGPT median tokens | 25 | `data/sharegpt_prompts.json` median = 25.0 | MATCH |
| ShareGPT p95 | 810 | data p95 = 810 | MATCH |
| ShareGPT clamp activation | ~36% | data fraction ≤16 = 36.42%, fraction <16 = 33.42% | MATCH |
| RTX 3060 12 GB | RTX 3060 12 GB | `SWEEP_METADATA.json` `gpu_name="NVIDIA GeForce RTX 3060"` | MATCH (12 GB SKU implicit) |

### Cross-section consistency (F3-F4)

- Variant names (`avmp_dynamic_b128`, `avmp_static_mr05`, `fixed_dual_mr05`, `fixed_dual_mr09`, `padded_unified`) consistent across all sections. MATCH.
- All 21 `\ref{...}` targets resolve to existing `\label{...}` in the section files. MATCH.
- All connector cap counts within target (Furthermore=1≤2, Conversely=2≤2, However=1≤8). MATCH per copy-edit pass 2.

---

## Phase 2 plan (after this AUDIT.md is committed)

Five MISMATCHes need decisions. The cheapest honest fixes:

| Mismatch | Recommended fix |
|---|---|
| M1 | Update §4.1, §1.2, §4.6, Abstract to say "180 cells, 2 pool sizes (1/4 GiB), models `jamba_1_5_mini` and `mamba2_1b3`". |
| M2 | Update §4.1 timedecomp parenthetical to "180 cells (60 aggregated × 3 seeds)" or just "180 cells". |
| M3 | Update §4.5 to say "298.67 MiB versus 336.00 MiB" (or round to "299 vs 336"). Re-derive from current aggregated.json. |
| M4 | Update §4.5 from "12.5%" to "11.1%". |
| M5 | Either (a) annotate the two-statistic gap with a one-sentence footnote, or (b) change prose point estimates to 12.93 / 2.19 / 1.83 (the bootstrap values). User decision needed. |

**Do NOT auto-fix M5 without confirmation**: this is a presentation/statistic decision that affects how the abstract reads (`13.3×` is the load-bearing headline). The MATCH inventory above shows the underlying source data and the CI bounds are correct; only the point-estimate-vs-CI alignment is the issue.

Phase 2 work proceeds per-mismatch with one commit each, each referencing the AUDIT.md line that motivates it.
