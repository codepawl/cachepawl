# Pre-Upload Checklist

This checklist is for manual review before any external preprint upload. It is
not an upload instruction and does not replace current arXiv rules.

## Package Build

- [ ] Build `paper.pdf` from inside this directory.
- [ ] Confirm `paper.bbl` is present and matches `references.bib`.
- [ ] Confirm the final PDF opens and has the expected title, abstract,
  tables, appendix, and bibliography.
- [ ] Confirm the final PDF does not contain unresolved references or
  unresolved citations.
- [ ] Confirm package files are limited to source files needed for the preprint
  and local documentation.

## Manual arXiv Checks

- [ ] Verify current arXiv source upload requirements.
- [ ] Verify accepted TeX engine and package support for the current source.
- [ ] Verify whether `paper.bbl`, `references.bib`, or both should be included.
- [ ] Verify file-extension, compression, and archive requirements.
- [ ] Verify category selection.
- [ ] Verify title, author metadata, affiliations, ORCID policy, and
  anonymization status.
- [ ] Verify abstract length and metadata formatting.
- [ ] Verify license selection and any repository/license compatibility notes.
- [ ] Verify whether ancillary files or artifact links are appropriate.

## Evidence Boundary

- [ ] Confirm the paper only claims advisory/planner-level diagnosis.
- [ ] Confirm it does not claim runtime allocator replacement.
- [ ] Confirm it does not claim runtime cache substitution.
- [ ] Confirm it does not claim live VRAM reduction.
- [ ] Confirm it does not claim throughput or latency improvement.
- [ ] Confirm it does not claim quality or accuracy improvement.
- [ ] Confirm controlled substitution remains future work until Mamba
  state-index and state tensor contracts are resolved.

## Repository Checks Before Upload

- [ ] Run the focused evidence tests listed in `README.md`.
- [ ] Confirm all evidence paths listed in `MANIFEST.md` exist.
- [ ] Regenerate the RTX 3060 planner-comparison pack into `/tmp` and diff it
  against the committed pack.
- [ ] Review `research/avmp/v2/evaluation/claim_summary.md`.
- [ ] Review `research/avmp/v2/paper/submission_plan.md`.

## Final Human Review

- [ ] Confirm authors and acknowledgments are final.
- [ ] Confirm no private, sensitive, or machine-local information is included.
- [ ] Confirm citations are intentional and complete enough for a preprint.
- [ ] Confirm the preprint route is still intended.
- [ ] Confirm no upload, push, tag, publish, or release action is performed by
  automation.
