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
