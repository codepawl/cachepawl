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

## 2026-05-23 — baseline measurement spine

- Added `cachepawl.bench` planner-probe package with JSONL result schema
- Added CPU-safe runtime/GPU environment capture without requiring CUDA or vLLM
- Added deterministic `short-heavy`, `long-heavy`, and `mixed` synthetic workloads
- Added RTX 3060 12GB benchmark config artifact
- Added `benchmarks/scripts/run_cache_probe.py` JSONL probe CLI
- Added tests for schema validation, serialization, environment fallbacks, deterministic workloads, and CLI output
- Verified: PawlKit check, ruff, format check, mypy, full pytest, and build
- Runtime vLLM serving, allocator replacement, monkeypatching, Triton kernels, and LSDR remain out of scope

## 2026-05-23 — planner comparison evidence

- Added vLLM-style padded cache-planning modeling baseline
- Added Cachepawl AVMP planner comparison path using the same synthetic workloads
- Added `benchmarks/scripts/compare_cache_planners.py` for JSONL records and Markdown/CSV summaries
- Added tests for planner behavior, schema round-trips, summary rendering, and deterministic CLI output
- Verified: PawlKit check, example comparison command, bench tests, ruff, format check, mypy, full pytest, and build
- Runtime vLLM serving, monkeypatching, allocator replacement, Triton kernels, copy kernels, LSDR, and real inference remain out of scope

## 2026-05-23 — planner metric semantics correction

- Replaced ambiguous `waste_ratio` benchmark fields with explicit `overestimation_ratio` and `wasted_fraction` fields
- Defined `overestimation_ratio` as `estimated_bytes / useful_bytes`
- Defined `wasted_fraction` as `(estimated_bytes - useful_bytes) / estimated_bytes`
- Updated JSONL records, Markdown/CSV summaries, planner estimates, synthetic probe output, and tests to use the corrected names
- Runtime vLLM serving, monkeypatching, allocator replacement, Triton kernels, copy kernels, LSDR, and real inference remain out of scope

## 2026-05-23 — planner comparison artifact pack

- Added a deterministic artifact-pack generator for the planner comparison
- Generated `benchmarks/results/rtx3060/planner-comparison/` with per-workload JSONL, combined summary, environment metadata, manifest, and reproduction README
- Added a narrow `.gitignore` allowlist for the RTX 3060 planner-comparison reference pack
- Added tests for deterministic artifact-pack generation
- Runtime vLLM serving, monkeypatching, allocator replacement, Triton kernels, copy kernels, LSDR, and real inference remain out of scope
