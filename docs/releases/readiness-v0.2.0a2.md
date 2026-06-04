# v0.2.0a2 Local Release Readiness

Date prepared: 2026-06-04

This record maps the current local release state to the advisory CLI alpha
checklist in `checklist-v0.x-advisory-alpha.md`.

## Target

- Package version: `0.2.0a2`
- Publish surface: `cachepawl diagnose-vllm`
- Release type: advisory CLI alpha
- Tag to use after approval: `v0.2.0a2`

## Completed Local Gates

- Version source of truth is `pyproject.toml`.
- `uv.lock` contains the same editable package version.
- `CHANGELOG.md` has an Advisory CLI Alpha entry.
- `scripts/check_version_consistency.py` can verify both local version
  consistency and tag/version consistency before a tag is pushed.
- `.github/workflows/publish.yml` is tag-gated, verifies the tag against the
  package version, runs the local quality gates, builds the package, and uses
  PyPI Trusted Publishing through the `pypi` environment.
- Runtime mutation remains disabled. The advisory CLI consumes artifacts and
  does not require vLLM, CUDA, GPU access, or NVML in the package environment.

## Release Boundary

The package is safe to tag and publish only as an advisory CLI alpha after the
local checklist commands pass. The tag/publish action itself is intentionally
not performed in this run.

Allowed claim:

- `cachepawl diagnose-vllm` turns translated vLLM cache-plan artifacts into
  advisory `report.json`, `summary.md`, and `manifest.json` outputs.

Disallowed claims:

- Runtime allocator replacement.
- Runtime cache substitution.
- Measured runtime VRAM reduction.
- Latency, throughput, quality, or accuracy improvement from the advisory CLI.

## Remaining External Gates Before Real Publish

- Release owner confirms `0.2.0a2` is the intended version.
- PyPI project `cachepawl` has Trusted Publishing configured for owner
  `codepawl`, repository `cachepawl`, workflow `publish.yml`, environment
  `pypi`.
- After explicit approval, create and push the `v0.2.0a2` tag.
- After the tag workflow completes, verify the PyPI sdist and wheel and run a
  fresh PyPI install smoke.
