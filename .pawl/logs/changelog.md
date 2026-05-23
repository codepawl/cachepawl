# Work Log

## 2026-05-23 — PawlKit setup for cachepawl

- Created `.pawl/` project operating folder
- Added active sprint tracker and task tracker
- Added decision log
- Added cachepawl-specific context files
- Added root `AGENTS.md` and Claude command/skill integration
- Completed Sprint 0 / T000 and opened Sprint 1 / T001 for vLLM integration
- Noted that `.agents/` and `.codex/` are read-only mounts in this workspace, so their optional adapter files could not be written here
- Verified PawlKit with `check` and `view`
- Verified repo checks: ruff, format check, mypy, pytest, and build
- No product code implemented

## 2026-05-23 — vLLM integration skeleton

- Added import-safe `cachepawl.integrations.vllm` planning skeleton
- Added frozen/slots cache-plan dataclasses and optional vLLM availability helpers
- Added focused tests under `tests/integration/vllm/`
- Updated README status wording to reflect implemented allocator prototypes and benchmark harnesses
- Verified: PawlKit check, ruff, format check, mypy, full pytest, and build
- Runtime vLLM serving, allocator replacement, monkeypatching, and Triton deployment remain out of scope
