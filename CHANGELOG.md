# Changelog

All notable user-facing changes for Cachepawl are tracked here.

Cachepawl is pre-alpha. Release notes distinguish advisory tooling from future
runtime mutation work.

## 0.2.0a1 - 2026-05-28

### Added

- Documented the advisory CLI alpha release checklist for the current
  `cachepawl diagnose-vllm` artifact-input workflow.
- Captured release verification commands for fresh install smoke, CLI help,
  existing artifact diagnosis, lint, format, type checking, tests, and build.
- Added `cachepawl diagnose-vllm` usability flags for summary stdout and
  advisory metric threshold exits.

### Notes

- Package version in `pyproject.toml`: `0.2.0a1`.
- Advisory CLI alpha release target confirmed.
- Runtime mutation remains intentionally disabled.
- vLLM is not a package dependency for the advisory CLI path.
