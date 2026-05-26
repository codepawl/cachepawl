# T007 — Mutation-readiness compatibility checks

Project: `.pawl/active/projects/project-main.md`
Sprint: `.pawl/active/sprints/sprint-005-mutation-readiness.md`
Status: Completed
Created: 2026-05-26
Updated: 2026-05-26
Completed: 2026-05-26
TTL: 30 days after completion or cancellation
Archive After: 2026-06-25
Archive Warning: 2026-06-18
Archive Reason: Completed mutation-readiness compatibility checks

## Objective

Implement a pre-mutation safety gate that checks compatibility and readiness
using the existing planner-stage and advisory-diff artifacts. This task does
not implement mutation.

## Inputs

- `research/avmp/v2/results/vllm-planner-stage-observation/translated_planner_stage_config.json`
- `research/avmp/v2/results/vllm-planner-stage-advisory-diff/diff_report.json`
- `research/avmp/v2/results/vllm-planner-stage-advisory-diff/group_level_diff.json`

## Expected Artifact

Write results under:

`research/avmp/v2/results/vllm-mutation-readiness/`

Expected files:

- `README.md`
- `manifest.json`
- `readiness_report.json`
- `summary.md`

## Result

Generated `research/avmp/v2/results/vllm-mutation-readiness/` with
`README.md`, `manifest.json`, `readiness_report.json`, and `summary.md`.

The readiness classification is `advisory_only_recommended`. The artifact
passed planner schema, `num_blocks`, cache group count, cache tensor count,
layer coverage, dtype/state dtype, block/page size, Mamba state shape,
attention/Mamba mapping, and estimated-byte consistency checks. It did not
fail structural invariants. It remains blocked from controlled substitution by
the mutation-required missing fields recorded in the T006 advisory diff:

- stable scheduler or planner construction hook
- allocator or KVCacheManager replacement control point
- worker tensor allocation layout control point
- runtime request-to-block assignment control
- Mamba state-index and attention view rewrite contract

## Required Checks

- planner output schema compatibility
- num blocks compatibility
- cache group count compatibility
- cache tensor count compatibility
- layer coverage compatibility
- dtype/state dtype compatibility
- block/page size compatibility
- Mamba state shape compatibility
- attention/Mamba group mapping compatibility
- estimated bytes/useful bytes consistency
- mutation-required missing fields

## Anti-Bypass Constraints

- Do not modify vLLM source.
- Do not monkeypatch vLLM.
- Do not return Cachepawl plans to vLLM.
- Do not replace allocators.
- Do not alter scheduler behavior.
- Do not alter worker tensor layout.
- Do not require vLLM, GPU, or NVML.
- Do not add Triton kernels, copy kernels, LSDR, serving changes, or quality
  evaluation.

## Done When

- [x] The T002 translated planner-stage config is consumed
- [x] The T006 diff report and group-level diff are consumed
- [x] `readiness_report.json`, `summary.md`, `manifest.json`, and `README.md`
  are generated
- [x] The artifact records non-mutation status
- [x] Missing mutation-required fields are reported
- [x] Focused tests or checks are recorded
- [x] PawlKit validation is recorded

## Progress Notes

- 2026-05-26: Implemented the import-safe mutation-readiness helper and
  benchmark artifact script.
- 2026-05-26: Generated the pre-mutation readiness artifact from T002 and T006
  inputs. The gate recommends advisory-only continuation until mutation
  control fields are designed and verified.

## Verification Log

2026-05-26 mutation-readiness implementation:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration/vllm/test_mutation_readiness.py tests/bench/test_vllm_mutation_readiness.py -q` — 5 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/create_vllm_mutation_readiness.py --timestamp 2026-05-26T00:00:00+00:00` — generated the artifact
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration/vllm -q` — 31 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_vllm_mutation_readiness.py -q` — 1 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 183 files already formatted
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py benchmarks/scripts/capture_vllm_cache_plan_observation.py benchmarks/scripts/capture_vllm_runtime_cache_plan_observation.py benchmarks/scripts/create_vllm_cache_diagnostic.py benchmarks/scripts/create_vllm_planner_dry_run_probe.py benchmarks/scripts/capture_vllm_planner_stage_observation.py benchmarks/scripts/create_vllm_planner_stage_advisory_diff.py benchmarks/scripts/create_vllm_mutation_readiness.py` — passed, 181 source files
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 403 passed, 12 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access; initial sandboxed attempt failed resolving `hatchling` from PyPI due DNS
- `git diff --check` — passed
- `npx @codepawl/pawlkit@0.3.0 view` — passed after approved npm access; initial sandboxed attempt failed with registry DNS `EAI_AGAIN`
- `npx @codepawl/pawlkit@0.3.0 check` — passed with 0 warnings after approved npm access; initial sandboxed attempt failed with registry DNS `EAI_AGAIN`
