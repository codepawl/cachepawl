# Workshop Package Manifest

## Package Contents

| Path | Role | Source |
| --- | --- | --- |
| `paper.tex` | Main two-column workshop manuscript with vector figures | This package |
| `supplement.tex` | Optional artifact, exact-value, and reproducibility supplement | This package |
| `references.bib` | Bibliography | `research/avmp/v2/paper/references.bib` |
| `paper.bbl` | Generated bibliography for the main paper | `bibtex paper` |
| `paper.pdf` | Compiled main workshop paper | `paper.tex`, `references.bib`, `paper.bbl` |
| `README.md` | Build and verification instructions | This package |
| `MANIFEST.md` | Package inventory | This package |
| `REVIEW_CHECKLIST.md` | Human review checklist | This package |

## Source-of-Truth Documents

All paths are relative to the repository root:

```text
research/avmp/v2/paper/draft.md
research/avmp/v2/paper/paper.tex
research/avmp/v2/paper/arxiv_package/paper.tex
research/avmp/v2/paper/submission_plan.md
research/avmp/v2/evaluation/claim_summary.md
research/avmp/v2/evaluation/README.md
benchmarks/results/rtx3060/planner-comparison/
```

## Evidence Artifacts

```text
research/avmp/v2/evaluation/matrix_table.md
research/avmp/v2/evaluation/matrix_table.csv
research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/report.json
research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/summary.md
research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/manifest.json
research/avmp/v2/results/vllm-planner-stage-observation/translated_planner_stage_config.json
research/avmp/v2/results/vllm-planner-stage-advisory-diff/diff_report.json
research/avmp/v2/results/vllm-planner-stage-advisory-diff/group_level_diff.json
benchmarks/results/rtx3060/planner-comparison/summary.md
benchmarks/results/rtx3060/planner-comparison/manifest.json
benchmarks/results/rtx3060/planner-comparison/environment.json
benchmarks/results/rtx3060/planner-comparison/short-heavy.jsonl
benchmarks/results/rtx3060/planner-comparison/long-heavy.jsonl
benchmarks/results/rtx3060/planner-comparison/mixed.jsonl
```

## Claim Boundary

Supported:

- advisory/planner-level diagnosis;
- bounded Path C matrix;
- deterministic RTX 3060 planner-only comparison;
- read-only artifact-input CLI.

Unsupported:

- runtime allocator replacement;
- runtime cache substitution;
- live VRAM reduction;
- throughput, latency, quality, or accuracy improvement;
- controlled substitution readiness before Mamba state-index and state tensor
  contracts are resolved.
