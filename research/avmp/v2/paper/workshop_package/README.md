# AVMP v2 Workshop Package

This directory contains a local workshop-style paper package for the AVMP v2
Path C evidence. It is prepared for review and local build only. Do not upload,
push, tag, publish, or release from this package.

## Contents

- `paper.tex`: two-column workshop-style manuscript with vector figures.
- `supplement.tex`: optional artifact and reproducibility supplement.
- `references.bib`: bibliography source.
- `paper.pdf`: compiled workshop-style paper.
- `paper.bbl`: generated bibliography file.
- `README.md`: local build instructions.
- `MANIFEST.md`: package contents and source mapping.
- `REVIEW_CHECKLIST.md`: human review checklist before submission.

## Build

From this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

Optional supplement build:

```bash
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
```

## Evidence Checks

From the repository root:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/bench/test_planner_comparison.py \
  tests/bench/test_vllm_path_c_advisory_matrix.py \
  tests/cli/test_diagnose_vllm.py -q
```

Regenerate the deterministic planner pack into `/tmp`:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  benchmarks/scripts/create_planner_comparison_pack.py \
  --output-dir /tmp/cachepawl-planner-comparison-workshop-check \
  --seed 1 --num-requests 128 \
  --gpu-name "NVIDIA GeForce RTX 3060" \
  --gpu-total-bytes 12884901888
```

Then diff the generated files against
`benchmarks/results/rtx3060/planner-comparison/`.

## Claim Boundary

This package supports advisory/planner-level diagnosis, the bounded Path C
matrix, the deterministic RTX 3060 planner-only comparison, and the read-only
artifact-input CLI. The main paper presents readable GiB-rounded tables and
figures; the supplement preserves exact byte values, paths, and commands.

It does not claim runtime allocator replacement, runtime cache substitution,
live VRAM reduction, throughput improvement, latency improvement, quality
improvement, accuracy improvement, or controlled substitution readiness.
