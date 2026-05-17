# AVMP paper (DRAFT)

**Skeleton only. No prose content written.** This folder ships the
LaTeX template, the build pipeline, the figure / table generation
scripts, and the bibliography seed for a paper on Asymmetric Virtual
Memory Paging (AVMP). The human author writes the prose into the files
under `sections/` later. This README documents the workflow.

## How to build the paper

From inside `research/avmp/`:

```bash
make all
```

This runs, in order:

1. `make figures`: regenerates the four figure files from the committed
   sweep JSONs.
2. `make tables`: regenerates the five table fragments.
3. `make class`: extracts `acmart.cls` (and supporting `.sty` files)
   from the vendored `acmart.dtx` via `latex acmart.ins`. One-time
   bootstrap; subsequent builds skip it because Make tracks the file.
4. `make paper`: `pdflatex` three-pass + `bibtex` to produce
   `paper.pdf`.

To regenerate only the figures or tables (e.g., after a fresh sweep):

```bash
make figures
make tables
```

To clean every generated artifact and rebuild from scratch:

```bash
make clean
make all
```

To produce an arXiv-ready archive once the paper is content-complete:

```bash
make arxiv
# arxiv-submission.tar.gz lands in the current directory
```

## Author workflow

1. Edit one section file under `sections/`. Each starts with a
   `CONTENT GUIDANCE` comment block citing the specific source (RFC,
   benchmarks results directory, or stage-N analysis markdown) the
   author should paraphrase from, plus `TODO author:` markers for each
   subsection body.
2. Run `make all`. The build pulls fresh data into the figures and
   tables before re-running `pdflatex`.
3. Review `paper.pdf`. Iterate on the section file.
4. When all sections are content-complete, delete the `CONTENT
   GUIDANCE` blocks and the `TODO author:` markers, then run `make
   arxiv` to produce `arxiv-submission.tar.gz`.

## Where the data comes from

Every figure and table is auto-generated from JSON committed under
`benchmarks/results/`:

- `scripts/generate_figures.py` reads
  `benchmarks/results/avmp-v2-batchsize-sweep/aggregated.json` and
  `benchmarks/results/avmp-v2-threshold-sweep/aggregated.json`.
- `scripts/generate_tables.py` reads the same two files, plus the live
  `AsymmetricVirtualPool.__init__` signature for
  `table_parameter_defaults.tex` (via `inspect.signature`).

No hardcoded numerical values appear in either script. If the
underlying sweep is regenerated (e.g., a follow-up run on different
hardware), `make all` reflects the new numbers without code edits.

## Known limitations

- **No mr09 in the headline figure.** No committed sweep covers both
  `fixed_dual_mr09` and `avmp_dynamic_b128` on identical cells, so
  `fig_oom_comparison_final.pdf` compares four variants
  (`padded_unified`, `fixed_dual_mr05`, `avmp_static_mr05`,
  `avmp_dynamic_b128`). If a future unified sweep adds mr09 to the
  same cell set, update `_HEADLINE_VARIANTS` in
  `scripts/generate_figures.py` and `scripts/generate_tables.py`.
- **State-machine diagram is hand-drawn.** This PR does NOT generate
  `figures/static/state-machine.pdf`. The author should draw it
  (e.g., in TikZ or Inkscape) and commit the PDF into
  `figures/static/`. The `Method` section's `CONTENT GUIDANCE` block
  references the expected path.
- **Header content is `Codepawl` / `codepawl@example.com`.** Replace
  before final submission.

## ACM template

This folder vendors `acmart` v2.16 (2025-08-27) from the official
release at <https://github.com/borisveytsman/acmart>, tag `v2.16`. The
class is shipped as the `.dtx` source plus `.ins` installer; `make
class` extracts `acmart.cls` on first build. SHA-256 of the vendored
files (verify before `make class` if you suspect tampering):

| file | SHA-256 |
|---|---|
| `acmart.dtx` | `cea16056a131d0c48f997a9c25524c1b9093cdc3d5d3a2bdcb8f2b56c3de4298` |
| `acmart.ins` | `e9715bc58dc26a164642d63398ddd7a8135d87ed7bc4ba10c30de40d02bfc514` |
| `ACM-Reference-Format.bst` | `ee0d9fd846b95ca8b9b7721e8d9aaa066c64c9e7a61284c6315cfbb844794a39` |

Document class options used in `paper.tex`:

```latex
\documentclass[sigconf, nonacm=true, anonymous=false, screen=true]{acmart}
```

- `sigconf`: ACM proceedings two-column layout (also used for arXiv
  pre-prints with `nonacm=true`).
- `nonacm=true`: drops the ACM rights block, since the paper has no
  ACM venue assignment yet.
- `anonymous=false`: not blind-review; the author byline appears.

If targeting MLSys, the template is different (see the MLSys CFP);
this folder will need a parallel `paper-mlsys.tex` driver.

## Build prerequisites

A full TeX Live install is the simplest route:

```bash
# Ubuntu / Debian
sudo apt install texlive-publishers texlive-fonts-recommended \
    texlive-latex-extra texlive-bibtex-extra biber
```

The Python figure / table generation only needs the dev dependencies
already in `pyproject.toml` (no new requirements).

## File layout

```
paper.tex                      # document entry point
sections/                      # section/subsection skeletons (no prose)
figures/
  static/                      # hand-drawn or external figures (committed)
  generated/                   # auto-generated from sweep JSONs (gitignored)
tables/
  generated/                   # auto-generated from sweep JSONs (gitignored)
bibliography/refs.bib          # 12 verified entries
scripts/
  generate_figures.py
  generate_tables.py
acmart.dtx                     # vendored source for acmart.cls
acmart.ins                     # installer that emits acmart.cls
ACM-Reference-Format.bst       # ACM bibliography style
Makefile                       # build targets
```
