# Changelog

All notable user-facing changes for Cachepawl are tracked here.

Cachepawl is pre-alpha. Release notes distinguish advisory tooling from future
runtime mutation work.

## Advisory CLI Alpha - 2026-05-28

### Added

- Documented the advisory CLI alpha release checklist for the current
  `cachepawl diagnose-vllm` artifact-input workflow.
- Captured release verification commands for fresh install smoke, CLI help,
  existing artifact diagnosis, lint, format, type checking, tests, and build.
- Added `cachepawl diagnose-vllm` usability flags for summary stdout and
  advisory metric threshold exits.

### Notes

- Package version is defined only in `pyproject.toml`.
- Advisory CLI alpha release target confirmed.
- Supersedes the unpublished `v0.2.0a1` tag for a clean trusted-publish run.
- Runtime mutation remains intentionally disabled.
- vLLM is not a package dependency for the advisory CLI path.
