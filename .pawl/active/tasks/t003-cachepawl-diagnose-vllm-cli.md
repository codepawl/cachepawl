# T003 — cachepawl diagnose-vllm CLI

Project: `.pawl/active/projects/project-main.md`
Sprint: `.pawl/active/sprints/sprint-002-planner-stage-observation.md`
Status: Completed
Created: 2026-05-26
Updated: 2026-05-26
Completed: 2026-05-26
TTL: 30 days after completion or cancellation
Archive After: 2026-06-25
Archive Warning: 2026-06-18
Archive Reason: Completed T003 artifact-input CLI productization

## Objective

Productize the existing vLLM observer/advisory value into a user-facing `cachepawl diagnose-vllm` command that can create diagnostic reports from existing runtime observation artifacts without requiring vLLM, GPU, or NVML in the main Cachepawl environment.

## Files Expected

- `src/cachepawl/` CLI entrypoint or existing command dispatch surface
- `cachepawl.integrations.vllm.translator`
- `cachepawl.integrations.vllm.observer`
- `cachepawl.integrations.vllm.advisory`
- `cachepawl.integrations.vllm.dry_run`
- Focused tests for artifact-input CLI behavior
- Optional docs or runbook updates if the command surface needs user-facing instructions

## Reproduction / Current Behavior

T001 produced successful observer/advisory artifacts and helper APIs, but the diagnostic is still exposed through benchmark/research scripts rather than a stable user-facing command. T002 planner-stage observation is blocked by host GPU/NVML access, but that blocker does not prevent productizing the already-successful artifact-based diagnostic path.

## Expected Behavior

The first CLI mode supports artifact input:

- read `translated_runtime_cache_config.json`
- read `raw_safe_metadata.json` when available
- emit `report.json`
- emit `summary.md`
- emit `manifest.json`

The command must be import-safe in the main Cachepawl environment when vLLM is not installed.

## Root Cause

The observer, advisory, dry-run, and artifact schemas exist as reusable building blocks, but they have not yet been assembled behind a package CLI intended for direct user diagnostics.

## Fix Strategy

- Add a minimal `cachepawl diagnose-vllm` command using the repository's existing CLI conventions.
- Implement artifact-input mode first, using existing serialized runtime observation artifacts rather than live vLLM objects.
- Reuse the existing translator, observer, advisory, dry-run, and runtime observation artifact format instead of introducing a second schema.
- Keep live vLLM probing out of the first CLI mode so the command is usable without vLLM, GPU, or NVML.
- Emit deterministic JSON and Markdown outputs suitable for committed artifacts and user inspection.

## Anti-Bypass Constraints

- Do not modify vLLM source.
- Do not monkeypatch vLLM.
- Do not replace allocators.
- Do not return any Cachepawl plan to vLLM.
- Do not require vLLM in the main Cachepawl environment for artifact-input mode.
- Do not require GPU or NVML for artifact-input mode.
- Do not add Triton kernels, copy kernels, LSDR, serving changes, or quality evaluation.
- Do not remove, skip, weaken, or fake tests/checks to make work pass.

## Steps

- [x] Identify the existing Cachepawl CLI entrypoint and command dispatch pattern.
- [x] Add artifact-input arguments for translated runtime config, optional raw metadata, and output directory or output files.
- [x] Load existing runtime observation artifacts through structured parsing.
- [x] Generate advisory report data using existing helper APIs.
- [x] Render `report.json`, `summary.md`, and `manifest.json`.
- [x] Add focused tests proving the command works without vLLM installed.
- [x] Record verification and update PawlKit trackers.

## Done When

- [x] `cachepawl diagnose-vllm` can run in the main Cachepawl environment without importing vLLM.
- [x] Artifact-input mode consumes `translated_runtime_cache_config.json` and optional `raw_safe_metadata.json`.
- [x] The command emits deterministic `report.json`, `summary.md`, and `manifest.json`.
- [x] Existing observer/advisory/dry-run concepts are reused rather than duplicated.
- [x] Focused tests cover the no-vLLM artifact-input path.
- [x] PawlKit validation passes.

## Commands To Run

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest <focused CLI tests> -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .`
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts`
- `npx @codepawl/pawlkit@0.3.0 view`
- `npx @codepawl/pawlkit@0.3.0 check`

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/cli/test_diagnose_vllm.py -q` — initially failed in the sandbox because uv could not fetch `hatchling>=1.25`; passed after approved PyPI access, 5 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run cachepawl diagnose-vllm --translated-cache-config research/avmp/v2/results/vllm-runtime-cache-plan-observation/translated_runtime_cache_config.json --raw-safe-metadata research/avmp/v2/results/vllm-runtime-cache-plan-observation/raw_safe_metadata.json --output-dir research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli --timestamp 2026-05-26T00:00:00+00:00` — passed and generated the committed example artifact
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration/vllm/test_planning.py tests/cli/test_diagnose_vllm.py -q` — 12 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 175 files already formatted
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py benchmarks/scripts/capture_vllm_cache_plan_observation.py benchmarks/scripts/capture_vllm_runtime_cache_plan_observation.py benchmarks/scripts/create_vllm_cache_diagnostic.py benchmarks/scripts/create_vllm_planner_dry_run_probe.py benchmarks/scripts/capture_vllm_planner_stage_observation.py` — passed, 173 source files
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 391 passed, 12 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed
- `npx @codepawl/pawlkit@0.3.0 view` — passed
- `npx @codepawl/pawlkit@0.3.0 check` — passed with 0 warnings

## Regression Coverage

Added focused CLI tests that run through `python -m cachepawl.cli diagnose-vllm` without vLLM installed and cover successful artifact generation, missing translated config file errors, invalid JSON errors, unsupported translated schema errors, and deterministic output for fixed inputs.

## Notes

- T002 remains blocked by host GPU/NVML access and should be rerun unchanged once GPU visibility is restored.
- T003 should not infer that `get_kv_cache_configs(...)` is unsafe or unavailable.
- The first CLI mode is an artifact-productization step, not a live vLLM integration or mutation step.
- Generated example artifact: `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/`.
- CLI command: `cachepawl diagnose-vllm --translated-cache-config <translated_runtime_cache_config.json> --raw-safe-metadata <raw_safe_metadata.json> --output-dir <output-dir>`.
