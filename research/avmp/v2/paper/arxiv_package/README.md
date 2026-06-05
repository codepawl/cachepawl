# AVMP v2 Local Preprint Package

This directory is a local, self-contained LaTeX preprint package for the
Cachepawl AVMP v2 Path C paper. It is prepared for local inspection and rebuild
only. Do not upload, publish, tag, or release from this directory without a
separate manual pre-upload review.

## Claim Boundary

The package preserves the advisory/planner-level claim boundary from the source
paper. It supports planner-level diagnosis, the bounded Path C advisory matrix,
the deterministic RTX 3060 planner-only comparison, and the read-only
artifact-input diagnostic CLI.

It does not claim runtime allocator replacement, runtime cache substitution,
live VRAM reduction, throughput improvement, latency improvement, quality
improvement, or accuracy improvement.

## Contents

- `paper.tex`: venue-neutral LaTeX source.
- `references.bib`: bibliography source.
- `paper.bbl`: generated bibliography file for upload-style builds.
- `paper.pdf`: compiled local preprint PDF.
- `MANIFEST.md`: package contents and source-of-truth mapping.
- `PRE_UPLOAD_CHECKLIST.md`: manual checklist before any external upload.

## Rebuild

From this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

The final PDF is written to:

```text
research/avmp/v2/paper/arxiv_package/paper.pdf
```

## Evidence Checks

From the repository root, the focused evidence checks are:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/bench/test_planner_comparison.py \
  tests/bench/test_vllm_path_c_advisory_matrix.py \
  tests/cli/test_diagnose_vllm.py -q
```

The planner-comparison pack can be regenerated with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  benchmarks/scripts/create_planner_comparison_pack.py \
  --output-dir /tmp/cachepawl-planner-comparison-arxiv-check \
  --seed 1 --num-requests 128 \
  --gpu-name "NVIDIA GeForce RTX 3060" \
  --gpu-total-bytes 12884901888
```

Then compare the generated files with:

```bash
diff -u benchmarks/results/rtx3060/planner-comparison/summary.md \
  /tmp/cachepawl-planner-comparison-arxiv-check/summary.md
diff -u benchmarks/results/rtx3060/planner-comparison/manifest.json \
  /tmp/cachepawl-planner-comparison-arxiv-check/manifest.json
diff -u benchmarks/results/rtx3060/planner-comparison/environment.json \
  /tmp/cachepawl-planner-comparison-arxiv-check/environment.json
diff -u benchmarks/results/rtx3060/planner-comparison/short-heavy.jsonl \
  /tmp/cachepawl-planner-comparison-arxiv-check/short-heavy.jsonl
diff -u benchmarks/results/rtx3060/planner-comparison/long-heavy.jsonl \
  /tmp/cachepawl-planner-comparison-arxiv-check/long-heavy.jsonl
diff -u benchmarks/results/rtx3060/planner-comparison/mixed.jsonl \
  /tmp/cachepawl-planner-comparison-arxiv-check/mixed.jsonl
```

## Manual Verification TODO

Current arXiv upload rules, TeX engine support, file-extension policies, source
archive expectations, and category guidance can change. Before uploading,
verify current arXiv instructions manually and update this package if needed.
