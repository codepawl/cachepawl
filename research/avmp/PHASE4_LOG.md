# Phase 4 Execution Log

- Started: 2026-05-20T12:53:17+00:00
- Branch: feat/paper-v15-phase4
- Base: main @ c3a0879
- Goal: V1.5 paper with paired bootstrap CI + phase-time decomposition + ShareGPT trace replay

## Task Status

- [ ] 4.3 Paired bootstrap CI on existing avmp-v2-throughput data
- [ ] 4.2 Goodput time decomposition (schema 1.2.0 to 1.3.0 + 90-cell CUDA sweep)
- [ ] 4.1 ShareGPT-Vicuna trace replay (30-cell CUDA sweep)

## Checkpoint log

(Entries appended as work progresses. Each entry must include an "Honesty note" line covering any claim the data forces us to soften.)

---

### Checkpoint 1: Bootstrap CI complete

- Timestamp: 2026-05-20T13:25:00+00:00
- Method: paired bootstrap, B=10000, RNG seed 20260520, ratio statistic = ratio of means
- Data: 36 paired observations cross-workload (12 per workload) from `benchmarks/results/avmp-v2-throughput/full/runs/`
- Reproducible: `uv run python research/avmp/scripts/bootstrap_ci.py`

Headline CI results:

| Claim | Point | 95% CI | Significant |
|---|---|---|---|
| cross-workload OOM delta (b128 vs mr05) | -3.50 / cell | [-5.83, -1.39] | yes |
| uniform_short OOM delta (b128 vs mr05) | +0.42 / cell | [0.00, 1.25] | **no** |
| mixed_long OOM delta (b128 vs mr05) | -5.75 / cell | [-10.3, -1.33] | yes |
| agentic_burst OOM delta (b128 vs mr05) | -5.17 / cell | [-9.25, -1.33] | yes |
| uniform_short goodput ratio (b128 / mr05) | 12.93x | [11.18, 16.00] | yes |
| mixed_long goodput ratio (b128 / mr05) | 2.19x | [1.70, 3.04] | yes |
| agentic_burst goodput ratio (b128 / mr05) | 1.83x | [1.42, 2.60] | yes |
| avmp_static vs mr05 (zero overhead) | 0.00 | [0.00, 0.00] | no (correctly null) |
| effective batch p50 delta (b128 - mr05) | 0.00 all workloads | [0.00, 0.00] | no (correctly null) |
| padded_unified vs mr05 OOM delta | +84.6 | [+46.2, +128] | yes |
| fixed_dual_mr09 vs mr05 OOM delta | +55.8 | [+29.2, +86.9] | yes |

V1 paper headline point estimates fall inside the bootstrap CIs (paper 13.30x in CI [11.18, 16.00]; paper 2.39x in CI [1.70, 3.04]; paper 1.83x exactly matches the bootstrap mean). The 7.6% cross-workload OOM reduction translates to per-cell CI [3.0%, 12.7%] relative to the mr05 baseline mean of 46.0 OOMs/cell, and both bounds are positive.

**Honesty note**: One previously-implicit caveat is now explicit in the paper. On `uniform_short`, AVMP records +0.42 OOMs/cell vs `fixed_dual_mr05` with CI [0.00, +1.25] - statistically inconclusive, not a win. The §4.2 figure caption already acknowledged the "within-1.7-OOM tie", but the prose did not distinguish per-workload outcomes from the cross-workload aggregate. §4.2 now explicitly states the per-workload split: AVMP wins where the workload shifts pressure between pools, and ties (within statistical noise) on the KV-only workload. No claims are softened away; the headline 13.3x and 7.6% remain because they sit inside their bootstrap CIs.

Files touched:
- NEW: `research/avmp/scripts/bootstrap_ci.py` (clean ruff + mypy --strict)
- NEW: `research/avmp/tables/generated/table_bootstrap_ci.tex`
- NEW: `research/avmp/tables/generated/bootstrap_ci.json`
- EDIT: `research/avmp/Makefile` (`make tables` invokes bootstrap_ci)
- EDIT: `research/avmp/sections/05-evaluation.tex` (§4.1 methodology, §4.2 CI annotation + per-workload nuance, §4.3 goodput CIs)

Build: PASS. paper.pdf at 9 pages.

Status: CHECKPOINT 1 COMPLETE - PROCEED to Task 4.2.

---

### Checkpoint 2: Time decomposition complete

- Timestamp: 2026-05-20T13:58:00+00:00
- Method: schema bump 1.2.0 -> 1.3.0 with 4 new int phase-time fields on `AllocatorMetrics`; runner instrumented to route allocate/free elapsed time into `time_in_service_ns` vs `time_in_oom_retry_ns` based on whether the call raised `OutOfMemoryError`; migration time bridged from `allocator_specific_stats["time_spent_rebalancing_ns"]`; idle is the residual.
- Sweep: 180 cells (5 variants x 3 workloads x 2 models x 2 pools x 3 seeds) on RTX 3060 12 GiB, 18:46 wall time, 0 failures.
- Output: `benchmarks/results/avmp-v15-timedecomp/`

Phase decomposition (median across model/pool cells, fractions of wall time):

| variant | workload | service | OOM retry | migration | idle |
|---|---|---|---|---|---|
| fixed_dual_mr05 | uniform_short | 99.16% | 0.01% | 0.00% | 0.83% |
| fixed_dual_mr05 | mixed_long | 72.60% | 26.34% | 0.00% | 1.06% |
| fixed_dual_mr05 | agentic_burst | 89.34% | 10.00% | 0.00% | 0.66% |
| avmp_dynamic_b128 | uniform_short | 87.54% | 0.04% | 0.00% | 12.42% |
| avmp_dynamic_b128 | mixed_long | 87.38% | 8.52% | 0.12% | 3.98% |
| avmp_dynamic_b128 | agentic_burst | 95.72% | 2.07% | 0.09% | 2.11% |

Mechanism check on absolute wall times for `uniform_short` at jamba_1_5_mini, 4 GiB:

| variant | goodput (req/s) | wall (s) | service (s) | OOM retry (s) |
|---|---|---|---|---|
| padded_unified | 984.1 | 0.50 | 0.36 | 0.00 |
| fixed_dual_mr05 | 15.8 | 32.37 | 32.19 | 0.00 |
| fixed_dual_mr09 | 37.9 | 13.61 | 13.24 | 0.16 |
| avmp_static_mr05 | 210.8 | 2.43 | 2.20 | 0.00 |
| avmp_dynamic_b128 | 221.0 | 2.32 | 2.10 | 0.00 |

**Honesty note - mechanism claim REFINED, not confirmed wholesale**:

V1 §4.3 attributed all of AVMP's goodput wins to "faster recovery from OOM events". The phase-time decomposition partially contradicts this:

1. **mixed_long and agentic_burst**: V1's mechanism IS confirmed. fixed_dual_mr05 burns 26.3% / 10.0% of wall time in OOM retry vs AVMP's 8.5% / 2.1%. Migration cost is <0.2% so the rebalancer pays for itself.
2. **uniform_short**: V1's mechanism is FALSIFIED. Both variants have ~0% OOM retry. The 13.30x goodput ratio is instead explained by raw per-call allocator service speed (fixed_dual_mr05 spends 32.2 s servicing 512 requests vs AVMP's 2.1 s at the same pool budget). The virtual-handle layer in avmp_static_mr05 closes most of this gap (2.2 s), so the speedup is attributable to the virtual addressing layer, not the dynamic rebalancer.

§4.3 has been rewritten honestly: it now states "two mechanisms, not one", and explicitly says the per-call speedup is an observation of the prototype implementation rather than a load-bearing design claim. The headline 13.30x / 2.39x / 1.83x ratios stay (they survive bootstrap CI) but the framing changes.

Files touched in Task 4.2:
- EDIT: `src/cachepawl/benchmarks/harness/metrics.py` (+4 fields + 2 collector methods)
- EDIT: `src/cachepawl/benchmarks/harness/schema.py` (SCHEMA_VERSION 1.3.0, _pop_int_with_default, additive serialization)
- EDIT: `src/cachepawl/benchmarks/harness/runner.py` (route elapsed to service vs oom_retry buckets)
- EDIT: `src/cachepawl/benchmarks/compare/aggregate.py` (4 _median fields on AggregatedRow)
- EDIT: `src/cachepawl/benchmarks/analysis/lexicographic_rank.py` (back-compat load with defaults)
- EDIT: `tests/unit/benchmarks/test_schema*.py`, `test_runner_smoke.py` + Aggregated row helper tests (4 new tests; schema-version assertions bumped; AggregatedRow positional helpers extended)
- EDIT: `benchmarks/README.md` (1.2.0 -> 1.3.0 migration row)
- EDIT: `research/avmp/Makefile` (`make figures` now also runs `generate_time_decomposition.py`)
- EDIT: `.gitignore` (allowlist `avmp-v15-timedecomp/`, gitignore its `runs/`)
- NEW: `research/avmp/scripts/generate_time_decomposition.py`
- NEW: `benchmarks/results/avmp-v15-timedecomp/aggregated.json` + `aggregated_deterministic.json` + `report.md` + `SWEEP_METADATA.json` + `figures/`
- NEW: `research/avmp/figures/generated/fig_time_decomposition.{pdf,png}` (regenerated by `make figures`)
- EDIT: `research/avmp/sections/05-evaluation.tex` (§4.3 mechanism paragraph + new figure include)

Build: PASS. paper.pdf at 10 pages (was 9 before this checkpoint).
Project gate: ruff + ruff format + mypy + 136 pytest tests all green.

Status: CHECKPOINT 2 COMPLETE with mechanism caveat. The headline goodput claims survive; the V1 mechanism narrative was too narrow and is now refined. PROCEED to Task 4.1.

---
