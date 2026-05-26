# T006 — Planner-stage post-call advisory diff artifact

Project: `.pawl/active/projects/project-main.md`
Sprint: `.pawl/active/sprints/sprint-004-planner-stage-advisory-diff.md`
Status: Completed
Created: 2026-05-26
Updated: 2026-05-26
Completed: 2026-05-26
TTL: 30 days after completion or cancellation
Archive After: 2026-06-25
Archive Warning: 2026-06-18
Archive Reason: Completed planner-stage post-call advisory diff artifact

## Objective

Compute and persist a Cachepawl proposed planner result beside the vanilla T002
planner-stage output as a post-call advisory/diff artifact, without returning
Cachepawl plans to vLLM or changing vanilla behavior.

## Starting Evidence

T005 completed the Path C mutation-hook design gate and D009 selected
planner-stage post-call advisory/diff as the safest next bounded experiment.
T002 produced the input artifact:

`research/avmp/v2/results/vllm-planner-stage-observation/translated_planner_stage_config.json`

The T002 replay reached real `vllm_config`, `kv_cache_specs`, and
`available_memory`; observed one worker, 63 specs, `FullAttentionSpec=9`,
`MambaSpec=54`, available memory `2915421184`, planner output
`num_blocks=329`, runtime scheduler `num_blocks=329`,
`planner_matches_runtime_scheduler=true`, and
`runtime_changed_during_replay=false`.

## Expected Artifact

Write results under:

`research/avmp/v2/results/vllm-planner-stage-advisory-diff/`

Expected files:

- `README.md`
- `manifest.json`
- `diff_report.json`
- `summary.md`
- `group_level_diff.json` if useful and derivable

## Result

Generated `research/avmp/v2/results/vllm-planner-stage-advisory-diff/` with
`README.md`, `manifest.json`, `diff_report.json`, `summary.md`, and
`group_level_diff.json`. The artifact records
`planner_stage_advisory_diff_available`, vanilla reserved bytes `2910781440`,
vanilla useful bytes `1679258112`, Cachepawl proposed reserved bytes
`1679258112`, estimated savings `1231523328`, overestimation ratio
`1.7333734577189286`, wasted fraction `0.4230902777777778`, 7 cache groups,
9 cache tensors, 63 layers, and 329 blocks.

## Required Comparisons

- vanilla planner reserved bytes
- vanilla useful bytes
- Cachepawl proposed reserved bytes
- estimated savings bytes
- overestimation ratio
- wasted fraction
- group-level differences where derivable
- missing fields that still prevent mutation
- parity and non-mutation status

## Anti-Bypass Constraints

- Do not return Cachepawl plans to vLLM.
- Do not modify vLLM source.
- Do not monkeypatch vLLM.
- Do not replace allocators.
- Do not alter scheduler behavior.
- Do not alter worker tensor layout.
- Do not add vLLM to the main Cachepawl dependencies.
- Do not add Triton kernels, copy kernels, LSDR, serving changes, or quality
  evaluation.

## Done When

- [x] The T002 translated planner-stage config is consumed
- [x] `diff_report.json`, `summary.md`, `manifest.json`, and `README.md` are
  generated
- [x] Group-level diff output is generated if derivable from available fields
- [x] The artifact records parity and non-mutation status
- [x] Missing mutation-blocking fields are reported
- [x] Focused tests or checks are recorded if product or script code changes
- [x] PawlKit validation is recorded

## Verification

Use the PawlKit commands in `.pawl/context/REPO_COMMANDS.md`. Product-code tests
are required only if T006 adds or changes product code.

## Progress Notes

- 2026-05-26: Opened T006 for the non-mutating planner-stage post-call
  advisory/diff artifact selected by D009.
- 2026-05-26: Validated the tracker opening with PawlKit. Initial sandboxed
  `npx @codepawl/pawlkit@0.3.0 view` and `check` attempts failed with npm
  registry DNS `EAI_AGAIN`; approved network reruns passed, and `check`
  reported 0 warnings.
- 2026-05-26: Implemented the import-safe planner-stage advisory diff helper,
  generated the non-mutating advisory/diff artifact from the T002 translated
  planner output, and kept vLLM behavior unchanged.

## Verification Log

2026-05-26 planner-stage advisory diff implementation:

- `UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/create_vllm_planner_stage_advisory_diff.py --timestamp 2026-05-26T00:00:00+00:00` — generated the artifact
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration/vllm -q` — 27 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_vllm_planner_stage_advisory_diff.py -q` — 1 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 179 files already formatted
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py benchmarks/scripts/capture_vllm_cache_plan_observation.py benchmarks/scripts/capture_vllm_runtime_cache_plan_observation.py benchmarks/scripts/create_vllm_cache_diagnostic.py benchmarks/scripts/create_vllm_planner_dry_run_probe.py benchmarks/scripts/capture_vllm_planner_stage_observation.py benchmarks/scripts/create_vllm_planner_stage_advisory_diff.py` — passed, 177 source files
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 398 passed, 12 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access; initial sandboxed attempt failed resolving `hatchling` from PyPI due DNS
- `npx @codepawl/pawlkit@0.3.0 view` — passed after approved npm access; initial sandboxed attempt failed with registry DNS `EAI_AGAIN`
- `npx @codepawl/pawlkit@0.3.0 check` — passed with 0 warnings after approved npm access; initial sandboxed attempt failed with registry DNS `EAI_AGAIN`
