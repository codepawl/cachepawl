# T004 — diagnose-vllm docs and smoke examples

Project: `.pawl/active/projects/project-main.md`
Sprint: `.pawl/active/sprints/sprint-002-planner-stage-observation.md`
Status: Completed
Created: 2026-05-26
Updated: 2026-05-26
Completed: 2026-05-26
TTL: 30 days after completion or cancellation
Archive After: 2026-06-25
Archive Warning: 2026-06-18
Archive Reason: Completed diagnose-vllm README documentation and smoke help check

## Objective

Document and polish the new `cachepawl diagnose-vllm` artifact-input CLI so it is release-ready for users who want a deterministic vLLM cache diagnostic without installing vLLM or requiring GPU/NVML access.

## Files Expected

- `README.md`
- Optional focused docs or smoke-test files if the existing README is too broad
- PawlKit tracker updates

## Reproduction / Current Behavior

T003 added the import-safe `cachepawl diagnose-vllm` CLI and generated `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/` with `report.json`, `summary.md`, and `manifest.json`. The command works from existing runtime observation artifacts, but the user-facing README does not yet explain the command, its no-vLLM/no-GPU mode, or its advisory-only safety boundary.

## Expected Behavior

The README includes a concise release-ready usage section for `cachepawl diagnose-vllm` that:

- shows an example command using `research/avmp/v2/results/vllm-runtime-cache-plan-observation/translated_runtime_cache_config.json`
- includes optional `raw_safe_metadata.json`
- points output at `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/`
- states that this CLI mode requires no vLLM dependency, CUDA, GPU, or NVML
- states that output is advisory-only and does not change vLLM behavior
- references generated `report.json`, `summary.md`, and `manifest.json`

## Root Cause

The CLI shipped before its README-facing usage and smoke-example text were added.

## Fix Strategy

- Add a small README usage snippet for the artifact-input diagnostic mode.
- Reuse the existing generated artifact path as the smoke example.
- Keep the text explicit about no runtime integration, no model loading, no vLLM dependency, no GPU/NVML requirement, and no vLLM behavior changes.
- Do not add new runtime integration or change the CLI behavior during this task.

## Anti-Bypass Constraints

- Do not continue T002.
- Do not modify vLLM source.
- Do not monkeypatch vLLM.
- Do not add vLLM to main Cachepawl dependencies.
- Do not add new runtime integration.
- Do not require GPU or NVML.
- Do not replace allocators.
- Do not add Triton kernels, copy kernels, LSDR, serving changes, or quality evaluation.
- Do not remove, skip, weaken, or fake checks to make work pass.

## Steps

- [x] Inspect current README structure and CLI artifact paths.
- [x] Add a concise `cachepawl diagnose-vllm` usage snippet.
- [x] Document the no-vLLM/no-GPU/no-NVML execution mode.
- [x] Document the advisory-only safety boundary and unchanged vLLM behavior.
- [x] Reference `report.json`, `summary.md`, and `manifest.json`.
- [x] Run focused docs/smoke verification plus PawlKit validation.

## Done When

- [x] README documents the `cachepawl diagnose-vllm` artifact-input command.
- [x] README includes an example using the existing runtime observation artifact.
- [x] README clearly states the command requires no vLLM, GPU, or NVML.
- [x] README clearly states outputs are advisory-only and do not change vLLM behavior.
- [x] README references generated `report.json`, `summary.md`, and `manifest.json`.
- [x] PawlKit validation passes.

## Commands To Run

- `npx @codepawl/pawlkit@0.3.0 view`
- `npx @codepawl/pawlkit@0.3.0 check`
- Optional focused smoke command if docs touch executable examples:
  `UV_CACHE_DIR=/tmp/uv-cache uv run cachepawl diagnose-vllm --translated-cache-config research/avmp/v2/results/vllm-runtime-cache-plan-observation/translated_runtime_cache_config.json --raw-safe-metadata research/avmp/v2/results/vllm-runtime-cache-plan-observation/raw_safe_metadata.json --output-dir /tmp/cachepawl-diagnose-vllm-smoke --timestamp 2026-05-26T00:00:00+00:00`

## Verification

- `cachepawl diagnose-vllm --help` — failed in this shell because the console script is not directly on `PATH`
- `UV_CACHE_DIR=/tmp/uv-cache uv run cachepawl diagnose-vllm --help` — passed and printed the documented `--translated-cache-config`, optional `--raw-safe-metadata`, `--output-dir`, and `--timestamp` arguments
- `npx @codepawl/pawlkit@0.3.0 view` — passed after approved npm access; initial sandboxed attempt failed with registry DNS `EAI_AGAIN`
- `npx @codepawl/pawlkit@0.3.0 check` — passed with 0 warnings after approved npm access; initial sandboxed attempt failed with registry DNS `EAI_AGAIN`

## Regression Coverage

The README command syntax was checked against `uv run cachepawl diagnose-vllm --help`.

## Notes

- T002 remains blocked by host GPU/NVML access and should not be continued in this task.
- T004 is documentation and release-readiness polish only.
- No product code, runtime integration, vLLM dependency, or vLLM behavior changed.
