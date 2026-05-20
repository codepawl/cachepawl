# AVMP paper figure & table formatting audit

Generated against `research/avmp/paper.pdf` at commit `e11d0fc` (vertical Fig 1, Fig 2 spacing landed on `feat/expand-citations`).

**Document class**: `\documentclass[sigconf, nonacm, screen]{acmart}`. All tables use `booktabs` rules. No `\cellcolor` used; staying monochrome.

**Total floats**: 10 (2 inline TikZ + 4 PDF figures + 4 generated tables that get `\input`). One additional generated table (`table_parameter_defaults.tex`) exists but is not `\input` anywhere; out of scope.

Legend for the checklist:

- `[x]` issue present, will be fixed in this PR
- `[!]` issue present, deferred (rationale stated)
- `[ ]` not applicable (qualitative figure, no obvious baseline, etc.)
- `[OK]` already passes

---

## Figure 1: AVMP virtual handle resolution

- Path: `sections/03-method.tex:29` `fig:page-table-structure`
- Type: inline TikZ block diagram (top-to-bottom flow)
- Metric: qualitative architecture diagram
- Direction: n/a
- Compares variants: no
- Issues:
  - [ ] Direction indicator — qualitative
  - [ ] Best/second-best — qualitative
  - [ ] Baseline delta — qualitative
  - [ ] Std notation — qualitative
  - [x] Math notation — `kv` and `ssm` strings in arrow labels could use the symbols defined in §3.1
  - [ ] Axis units — n/a
  - [x] Self-contained caption — caption is mostly self-contained; minor polish in rule 7 commit
  - [ ] Numerical precision — n/a
  - [ ] Alignment — n/a
- Proposed fix: light caption polish; no structural change.

## Figure 2: Pool rebalancing state machine

- Path: `sections/03-method.tex:167` `fig:state-machine`
- Type: inline TikZ FSM diagram
- Metric: qualitative
- Direction: n/a
- Compares variants: no
- Issues:
  - [ ] Direction indicator — qualitative
  - [ ] Best/second-best — qualitative
  - [ ] Baseline delta — qualitative
  - [ ] Std notation — qualitative
  - [x] Math notation — edge labels reference `th_low` and `th_high` config keys; keep as code-identifier (teletype), no math substitution
  - [ ] Axis units — n/a
  - [x] Self-contained caption — add a sentence explaining edge-label semantics and `CapacityError` trigger placement
  - [ ] Numerical precision — n/a
  - [ ] Alignment — n/a
- Proposed fix: caption rewrite per rule 7 template.

## Table 1: Cross-workload OOM baseline comparison

- Path: `sections/05-evaluation.tex:18` `tab:baseline-comparison`
- Source: `tables/generated/table_baseline_comparison.tex` (built by `scripts/generate_tables.py::table_baseline_comparison`)
- Type: 4-variant × 4-column OOM data table
- Metric: `oom_count_mean` summed across 12 cells per (variant, workload)
- Direction: ↓ lower is better (caption-level)
- Compares variants: yes (4 allocator variants)
- Baseline row: `padded_unified` (first row, the unmodified vLLM-style unified pool)
- Issues:
  - [x] Direction indicator — caption-level `(↓)` for OOM
  - [x] Best/second-best — apply `\textbf{}` and `\underline{}` per column
  - [x] Baseline delta — add `Δ% vs padded_unified` column on the `Total`
  - [x] Std notation — surface `oom_count_std` via `\sqrt{\sum \mathrm{var}_i}` propagation across the 12 summed cells; format `mean ± std` per cell
  - [x] Math notation — caption refers to `$N_{\mathrm{OOM}}$`
  - [ ] Axis units — n/a (table)
  - [x] Self-contained caption — rule 7 rewrite
  - [x] Numerical precision — keep `.1f` on means; std `.1f`; delta `.1f%`
  - [OK] Alignment — `lrrrr` already right-aligns numerics
- Proposed fix: per-column highlight + Δ% column + mean ± std + caption rewrite.

## Figure 3: Cross-allocator OOM comparison (final)

- Path: `sections/05-evaluation.tex:25` `fig:oom-comparison-final`
- Source: `figures/generated/fig_oom_comparison_final.pdf` (built by `scripts/generate_figures.py::fig_oom_comparison_final`)
- Type: grouped bar chart, 4 variants × 3 workloads
- Metric: OOM count, summed across 12 cells
- Direction: ↓
- Compares variants: yes
- Issues:
  - [x] Direction indicator — caption-level `(↓)`
  - [!] Best-bar highlighting — DEFERRED: hatching/stroke on the lowest bar per workload would clutter the legend; reader can read off lowest visually. Bar palette already colorblind-safe (ColorBrewer Set2).
  - [ ] Baseline delta — implicit via grouping; not a redundant column
  - [!] Std error bars — DEFERRED: would need propagation across 12 cells per group; the prose already cites cross-cell std `(0.8 to 3.0)`. Adding error bars to 12 stacked groups risks visual clutter; deferred to follow-up if reviewer requests
  - [x] Math notation — caption uses `$N_{\mathrm{OOM}}$`
  - [x] Axis units — y-axis label currently `Sum of mean OOMs across 12 cells` (internal jargon); change to `Total OOM events` (script edit, regenerate PDF)
  - [x] Self-contained caption — rule 7 rewrite
  - [OK] Numerical precision — bars are visual; tick labels matplotlib default
  - [OK] Alignment — n/a
- Proposed fix: script edits y-axis label and title; caption rewrite.

## Table 2: Per-workload goodput winner

- Path: `sections/05-evaluation.tex:43` `tab:per-workload-winner`
- Source: `tables/generated/table_per_workload_winner.tex` (built by `scripts/generate_tables.py::table_per_workload_winner`)
- Type: 2-variant × 3-workload goodput + ratio
- Metric: `goodput_requests_per_second_median` (req/s)
- Direction: ↑ higher is better
- Compares variants: yes (2 variants)
- Baseline column: `fixed_dual_mr05`
- Issues:
  - [x] Direction indicator — header-level `(req/s, ↑)` on goodput columns; `Ratio` plain
  - [x] Best/second-best — 2 variants → bold the winner column value per row (no second-best with only 2 variants)
  - [x] Baseline delta — ratio column already encodes this; caption defines `Ratio = avmp_dynamic_b128 / fixed_dual_mr05`
  - [!] Std notation — DEFERRED: goodput std not exposed by `aggregate.py`. Flag in caption: "goodput std not surfaced by current pipeline; values are point estimates". Pipeline change is out of scope.
  - [x] Math notation — caption refers goodput as `g` (req/s) and ratio as `r = g_{\mathrm{AVMP}} / g_{\mathrm{baseline}}`
  - [ ] Axis units — n/a
  - [x] Self-contained caption — rule 7 rewrite
  - [OK] Numerical precision — `.2f` goodput, `.2f×` ratio; document caption
  - [OK] Alignment — `lrrr`
- Proposed fix: header direction, bold winner per row, caption rewrite, flag missing std.

## Figure 4: OOMs vs migration_batch_size

- Path: `sections/05-evaluation.tex:61` `fig:oom-vs-batch-size`
- Source: `figures/generated/fig_oom_vs_batch_size.pdf` (built by `scripts/generate_figures.py::fig_oom_vs_batch_size`)
- Type: line plot, 3 workload series × 9 B values
- Metric: OOM count summed across 12 cells
- Direction: ↓
- Compares variants: implicit (9 B values; dotted lines mark `fixed_dual_mr05` baseline)
- Issues:
  - [x] Direction indicator — caption-level `(↓)`
  - [!] Best-marker highlighting — DEFERRED: 3 lines × 9 points = 27 points; marking minima per line adds noise. The narrative text identifies `b128` and `b256` as joint near-optima.
  - [x] Baseline delta — already drawn as dotted reference line; mention in caption legend
  - [!] Std notation — DEFERRED: same propagation rationale as Figure 3
  - [x] Math notation — x-axis label refers `B` (migration batch size); use the symbol
  - [x] Axis units — x-axis `migration_batch_size` → `Migration batch size B (log_2 scale)`; y-axis `Sum of mean OOMs across 12 cells` → `Total OOM events`
  - [x] Self-contained caption — rule 7 rewrite
  - [OK] Numerical precision — n/a (visual)
  - [OK] Alignment — n/a
- Proposed fix: script edits axis labels; caption rewrite.

## Table 4: Stage 1 migration batch size sweep

- Path: `sections/05-evaluation.tex:68` `tab:stage1-batchsize`
- Source: `tables/generated/table_stage1_batchsize.tex` (built by `scripts/generate_tables.py::table_stage1_batchsize`)
- Type: 9-row B-sweep × 4-column OOM
- Metric: OOM count summed across 12 cells per (B, workload)
- Direction: ↓ caption-level
- Compares variants: yes (9 B values; pure parameter sweep, no obvious baseline row)
- Issues:
  - [x] Direction indicator — caption-level `(↓)`
  - [x] Best/second-best — `\textbf{}` per column on minimum, `\underline{}` on second-smallest
  - [ ] Baseline delta — parameter sweep, no canonical baseline
  - [x] Std notation — surface `oom_count_std` via propagation; format `mean ± std`
  - [x] Math notation — header `batch_size` → `B` (with `B` defined in §3.1)
  - [ ] Axis units — n/a
  - [x] Self-contained caption — rule 7 rewrite
  - [x] Numerical precision — `.1f` means, `.1f` std
  - [OK] Alignment — `rrrrr`
- Proposed fix: highlight + std + caption rewrite + `B` header.

## Figure 5: Threshold sensitivity

- Path: `sections/05-evaluation.tex:77` `fig:threshold-sensitivity`
- Source: `figures/generated/fig_threshold_sensitivity.pdf` (built by `scripts/generate_figures.py::fig_threshold_sensitivity`)
- Type: bar chart, 5 threshold variants (b128 reference + 4 sweep)
- Metric: cross-workload total OOM
- Direction: ↓
- Compares variants: yes; all tie at 510 (null result)
- Issues:
  - [x] Direction indicator — caption-level `(↓)`
  - [ ] Best highlighting — all variants tie at 510.0; no winner to bold
  - [ ] Baseline delta — sweep is the experimental message; no baseline row
  - [!] Std notation — DEFERRED, same propagation rationale
  - [x] Math notation — caption refers `N_{\mathrm{OOM}}`
  - [x] Axis units — y-axis `Cross-workload total OOMs (12 cells x 3 workloads)` → `Total OOM events`; cell count moves to caption
  - [x] Self-contained caption — rule 7 rewrite, narrate null result
  - [OK] Numerical precision — bar value labels `.1f`
  - [OK] Alignment — n/a
- Proposed fix: script edits y-axis label; caption rewrite.

## Table 5: Stage 2 threshold sweep

- Path: `sections/05-evaluation.tex:86` `tab:stage2-threshold`
- Source: `tables/generated/table_stage2_threshold.tex` (built by `scripts/generate_tables.py::table_stage2_threshold`)
- Type: 5-row threshold-sweep × 4-column (low, high, total_oom, rebalance_count)
- Metric: mixed (thresholds = ratios, total_oom = ↓, rebalance_count = descriptive)
- Direction: per-column (mixed)
- Compares variants: yes (5 rows); all tie on `total_oom` and `rebalance_count`
- Baseline row: `avmp_dynamic_b128` (first row, the default; rest are perturbations)
- Issues:
  - [x] Direction indicator — per-column header: `total_oom (↓)`, others plain
  - [ ] Best/second-best — all tied at 510.0 / 84; caption already states tie. No highlighting needed
  - [ ] Baseline delta — all rows identical; delta would be all zeros
  - [x] Std notation — surface `oom_count_std` per row; the tie at means doesn't imply tied stds
  - [ ] Math notation — `threshold_low`, `threshold_high`, `rebalance_count` are config-field references; keep teletype
  - [ ] Axis units — n/a
  - [x] Self-contained caption — rule 7 rewrite, narrate null result clearly
  - [x] Numerical precision — keep `.2f` thresholds, `.1f` OOM, `.0f` rebalance count; explain mixed precision in caption
  - [OK] Alignment — `lrrrr`
- Proposed fix: per-column direction in header + std + caption rewrite.

## Figure 6: Peak reserved VRAM trade-off

- Path: `sections/05-evaluation.tex:100` `fig:peak-reserved-tradeoff`
- Source: `figures/generated/fig_peak_reserved_tradeoff.pdf` (built by `scripts/generate_figures.py::fig_peak_reserved_tradeoff`)
- Type: bar chart, 4 variants
- Metric: peak reserved VRAM in MiB
- Direction: ↓ (memory overhead)
- Compares variants: yes (4 variants)
- Issues:
  - [x] Direction indicator — caption-level `(↓ less overhead is better; AVMP intentionally larger)`
  - [!] Best-bar highlighting — DEFERRED: the smallest bar is `padded_unified` which is the WORST allocator otherwise. Highlighting it as "best on this metric" misleads. Caption narrates the trade-off
  - [ ] Baseline delta — no clean baseline; the 2× overhead claim is in the prose
  - [x] Std notation — surface `peak_reserved_bytes_std` as error bars + value labels `mean ± std`
  - [x] Math notation — y-axis already in MiB; no further math
  - [x] Axis units — y-axis `Mean peak_reserved (MiB)` → `Peak reserved VRAM (MiB)`
  - [x] Self-contained caption — rule 7 rewrite
  - [OK] Numerical precision — bar labels `,.0f` (1000s separator); keep
  - [OK] Alignment — n/a
- Proposed fix: script: error bars + cleaner y-axis label; caption: rewrite + ± std.

---

## Cross-cutting decisions

- **Best-bar highlighting on figures (rule 2)**: deferred for all 4 PDF figures. Bar charts already make minima visible; hatching introduces print-style ambiguity in monochrome and clutters the legend in color. Document explicitly here so the reviewer sees the conscious decision rather than an oversight.
- **Goodput std (Table 2 rule 4)**: flagged as missing pipeline data. Adding it requires modifying `cachepawl.benchmarks.compare.aggregate` to expose `goodput_*_std`. Out of scope for a docs-only PR; will open a follow-up issue.
- **Std propagation across summed cells**: `aggregate.py` exposes per-cell `oom_count_std` and `peak_reserved_bytes_std`. Tables 1/4/5 sum across cells; the std of a sum under independence is `\sqrt{\sum \mathrm{var}_i}`. We adopt this propagation (standard error propagation, not data fabrication) and state the formula in the relevant captions.
- **Significance markers (`*`, `**`, `***`)**: skipped. The paper does not run hypothesis tests; spec forbids fabrication.
- **Color-coded cells**: skipped. `acmart` monochrome convention; no `\cellcolor`.
- **Math notation scope**: `N_{\mathrm{OOM}}` (OOM event count), `B` (migration batch size), `g` (goodput, req/s), `r` (ratio between two goodput values). Defined on first use in body text.
- **Captions in rule-7 template**: each caption ends with one sentence of substantive observation; no forward references.

## Fix count per rule (target)

| Rule | Description | Floats affected |
|------|-------------|-----------------|
| 1 | Direction indicators | 8 (all quantitative) |
| 2 | Best/second-best highlight | 3 tables (T1, T2, T4) |
| 3 | Baseline Δ% | 1 table (T1) |
| 4 | Mean ± std | 3 tables (T1, T4, T5) + 1 fig (F6) |
| 5 | Math notation | §3.1 prose + 6 captions |
| 6 | Axis labels / units | 4 figures (script edits) |
| 7 | Self-contained captions | all 10 floats |
| 8 | Numerical precision | 0 (already consistent per-column) |
| 9 | Alignment | 0 (already correct) |
