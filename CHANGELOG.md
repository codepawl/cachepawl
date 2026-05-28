# Changelog

All notable user-facing changes for Cachepawl are tracked here.

Cachepawl is pre-alpha. Release notes distinguish advisory tooling from future
runtime mutation work.

## Unreleased

### Added

- Documented the advisory CLI alpha release checklist for the current
  `cachepawl diagnose-vllm` artifact-input workflow.
- Captured release verification commands for fresh install smoke, CLI help,
  existing artifact diagnosis, lint, format, type checking, tests, and build.
- Added `cachepawl diagnose-vllm` usability flags for summary stdout and
  advisory metric threshold exits.

### Notes

- Current package version in `pyproject.toml`: `0.1.0`.
- Recommended next advisory alpha version: `0.2.0a1`.
- Runtime mutation remains intentionally disabled.
- vLLM is not a package dependency for the advisory CLI path.
