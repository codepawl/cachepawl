# Workshop Review Checklist

## Paper Presentation

- [ ] Confirm the compiled PDF looks like a workshop paper rather than a
  repository report.
- [ ] Confirm the title is concise and research-facing.
- [ ] Confirm the anonymous author block matches the intended review policy.
- [ ] Confirm the abstract is 150-200 words and follows problem, method,
  evidence, limitation.
- [ ] Confirm main-paper tables are readable in two-column format.
- [ ] Confirm path-heavy reproducibility material stays in the supplement.

## Claim Boundary

- [ ] Confirm the paper claims only advisory/planner-level diagnosis.
- [ ] Confirm the paper does not claim runtime allocator replacement.
- [ ] Confirm the paper does not claim runtime cache substitution.
- [ ] Confirm the paper does not claim live VRAM reduction.
- [ ] Confirm the paper does not claim throughput or latency improvement.
- [ ] Confirm the paper does not claim quality or accuracy improvement.
- [ ] Confirm controlled substitution remains future work until Mamba
  state-index and state tensor contracts are resolved.

## Evidence Checks

- [ ] Build `paper.pdf` from inside this directory.
- [ ] Build `supplement.pdf` if submitting supplemental material.
- [ ] Confirm no unresolved citations or references in the final log.
- [ ] Confirm no overfull boxes in the final log.
- [ ] Run the focused evidence tests listed in `README.md`.
- [ ] Regenerate the RTX 3060 planner pack into `/tmp` and diff it against the
  committed artifacts.
- [ ] Confirm all artifact paths listed in `MANIFEST.md` exist.

## Venue Checks

- [ ] Select the actual workshop template if required.
- [ ] Apply page limit and anonymity rules for the target workshop.
- [ ] Confirm whether supplement submission is allowed.
- [ ] Confirm bibliography style and formatting rules.
- [ ] Confirm author metadata, acknowledgments, license, and artifact policy
  after the review policy is known.
