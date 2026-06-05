# Submission Route Plan

## Current category

arXiv/preprint-ready.

## Rationale

The current package is suitable as a preprint or technical report because it is buildable, venue-neutral, and aligned to the committed evidence boundary. The manuscript states advisory and planner-level claims, includes compact tables for the Path C matrix and RTX 3060 planner-only comparison, and separates supported claims from unsupported runtime claims.

The package is not yet workshop/short-paper-ready because no target template, page budget, anonymization rule, or venue-specific artifact format has been selected. It is also still framed as a venue-neutral manuscript rather than a tightly scoped short-paper submission.

The package is not full-conference-ready because the committed evidence does not measure runtime allocator replacement, runtime cache substitution, live VRAM reduction, throughput, latency, quality, or accuracy. Controlled substitution readiness also remains blocked until Mamba state-index and state tensor contracts are resolved.

## Current supported package

- Markdown source reference: `research/avmp/v2/paper/draft.md`
- LaTeX manuscript: `research/avmp/v2/paper/paper.tex`
- Bibliography placeholders: `research/avmp/v2/paper/references.bib`
- Claim boundary: `research/avmp/v2/evaluation/claim_summary.md`
- Evaluation notes: `research/avmp/v2/evaluation/README.md`
- Planner-only artifacts: `benchmarks/results/rtx3060/planner-comparison/`

## Claims to preserve

- Planner-level over-reservation exists in the bounded Path C Zyphra/Zamba2-2.7B-instruct matrix.
- RTX 3060 planner-only evidence shows `cachepawl-avmp` stays closer to useful bytes than the padded planner baseline.
- The current package supports advisory artifact-input diagnosis.
- The tool is read-only and non-mutating.

## Claims outside the current boundary

- Runtime allocator replacement.
- Runtime cache substitution.
- Live VRAM reduction.
- Throughput, latency, quality, or accuracy improvement.
- Controlled substitution readiness before Mamba state-index and state tensor contracts are resolved.

## Smallest upgrade to workshop/short-paper-ready

- Select a target workshop or short-paper format and apply its template, page limit, anonymization policy, and artifact rules.
- Compress the manuscript to the selected page budget without expanding the claim boundary.
- Move the artifact and reproducibility checklist to the appendix or supplement if the selected format requires it.
- Add a small bounded evidence strengthening step, such as an additional planner-only or advisory diagnostic replication, only if it can be committed without implying runtime substitution or live serving performance.
- Re-run the LaTeX build, artifact path checks, and evidence tests after formatting.

## Evidence needed beyond workshop/short-paper readiness

- Resolved Mamba state-index and state tensor contracts for a controlled substitution probe.
- Default-off substitution experiment with parity and rollback checks.
- Live serving measurements for VRAM behavior and request admission if those claims are made.
- Throughput and latency measurements if performance claims are made.
- Multi-model and multi-workload evidence before generalizing beyond the bounded current matrix.
- Quality or accuracy measurements if output-quality claims are made.

## Formatting TODO

The current LaTeX manuscript intentionally remains venue-neutral. Venue selection should happen before changing document class, bibliography style, page limits, anonymity settings, or artifact checklist placement.
