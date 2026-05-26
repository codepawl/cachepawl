# T002 — Real planner-stage observation around get_kv_cache_configs

Project: `.pawl/active/projects/project-main.md`
Sprint: `.pawl/active/sprints/sprint-002-planner-stage-observation.md`
Status: Blocked
Created: 2026-05-25
Updated: 2026-05-26
Completed: N/A
TTL: 30 days after completion or cancellation
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Objective

Determine whether Cachepawl can safely observe or call vLLM 0.21.0 planner-stage cache planning around `get_kv_cache_configs(...)` with real `VllmConfig`, `dict[str, KVCacheSpec]`, and available-memory inputs, without mutating vLLM behavior.

## Blocker

This task is blocked by host GPU/NVML access in the current host/session. CUDA is unavailable in both the pinned vLLM environment and the main uv environment, and `nvidia-smi` reports GPU access blocked by the operating system. This is not evidence that `get_kv_cache_configs(...)` is unsafe or unavailable; it means the real runtime objects needed for the planner-stage call boundary cannot be reached in this session.

## Current Behavior

T001 closed the observe-first runtime boundary. Cachepawl can translate real vLLM cache planning dataclasses, observe `LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config`, emit advisory diagnostics, and create a non-mutating planner dry-run artifact. The actual planner-stage `get_kv_cache_configs(...)` call boundary has not yet been reached with real runtime inputs.

## Expected Behavior

The repo records a bounded planner-stage observation artifact showing either:

- direct real planner-stage observation or translation around `get_kv_cache_configs(...)`; or
- a structured blocker explaining the exact private/runtime-only input or unstable API boundary that prevents safe observation.

## Strategy

- Use `/tmp/vllm-cachepawl-venv` with pinned `vllm==0.21.0`.
- Use `PYTHONPATH=src` so Cachepawl imports from the workspace without adding vLLM to the main environment.
- Start from the already-proven runtime path `LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config`.
- Inspect only the minimum real vLLM objects needed to identify `VllmConfig`, per-worker `KVCacheSpec` maps, and available-memory inputs.
- Attempt a read-only call or observation around `get_kv_cache_configs(...)` only if inputs can be obtained without unsafe reconstruction.
- Translate any reached cache config through `cachepawl.integrations.vllm.translator`.
- If private internals or unsafe runtime state are required, produce a structured blocker artifact instead of widening scope.

## Anti-Bypass Constraints

- Do not modify vLLM source.
- Do not monkeypatch vLLM.
- Do not return any Cachepawl plan to vLLM.
- Do not replace allocators.
- Do not alter scheduler behavior.
- Do not alter worker tensor layout.
- Do not add vLLM to the main Cachepawl dependencies.
- Do not add Triton kernels, copy kernels, LSDR, serving changes, or quality evaluation.
- Do not run long-lived `vllm serve`.

## Done When

- [x] The planner-stage observation attempt is bounded and reproducible
- [x] The artifact directory `research/avmp/v2/results/vllm-planner-stage-observation/` contains `README.md` and `manifest.json`
- [ ] The artifact contains translated planner-stage output if real objects are reached
- [x] The artifact contains `blocker.json` if safe planner-stage access is blocked
- [ ] The result compares planner-stage fields against the runtime observer and dry-run assumptions
- [x] PawlKit validation is recorded
- [x] Focused tests or checks are recorded if code is added or changed

## Verification

Use the commands in `.pawl/context/REPO_COMMANDS.md`. Product-code tests are required only if T002 adds or changes product code.

## Progress Notes

- 2026-05-25: Opened T002 after T001 closed the observe-first vLLM integration boundary. This task is limited to real planner-stage observation around `get_kv_cache_configs(...)` and must remain non-mutating.
- 2026-05-25: Accepted D008 to capture PawlKit issues only when normal
  Cachepawl work exposes real friction. No product code or research direction
  changed.
- 2026-05-25: Added
  `benchmarks/scripts/capture_vllm_planner_stage_observation.py`. The script
  is import-safe in the main Cachepawl environment, uses the pinned vLLM
  environment when available, and only calls
  `vllm.v1.core.kv_cache_utils.get_kv_cache_configs(...)` on deep-copied real
  inputs. It never returns a Cachepawl plan to vLLM and does not mutate vLLM
  source, scheduler behavior, allocators, or worker tensor layout.
- 2026-05-25: Added
  `tests/bench/test_vllm_planner_stage_observation.py` for the blocker artifact
  path when vLLM is unavailable in the main environment.
- 2026-05-25: Ran the planner-stage observation script through
  `/tmp/vllm-cachepawl-venv` with `PYTHONPATH=src`. The run produced
  `research/avmp/v2/results/vllm-planner-stage-observation/` with
  `README.md`, `manifest.json`, `blocker.json`, and `raw_safe_metadata.json`.
  The artifact is blocked because torch reports CUDA unavailable in the pinned
  vLLM environment during this session. Therefore `get_kv_cache_configs(...)`
  was not called, and real runtime `VllmConfig`, `KVCacheSpec` dictionaries,
  and available-memory inputs were not reached.
- 2026-05-25: The blocker artifact still records safe static metadata from
  importable vLLM 0.21.0: `get_kv_cache_configs(...)` has signature
  `(vllm_config, kv_cache_specs, available_memory)`, and the source call site is
  `vllm.v1.engine.core.EngineCore._initialize_kv_caches`, which obtains
  `kv_cache_specs` from `self.model_executor.get_kv_cache_specs()` and
  available memory from `self.model_executor.determine_available_memory()` with
  `self.available_gpu_memory_for_kv_cache` retained afterward.
- 2026-05-25: Stabilized the blocker diagnosis after checking the pinned vLLM
  env, main uv env, and host GPU state. `/tmp/vllm-cachepawl-venv` has
  torch `2.11.0+cu130`, CUDA `13.0`, CUDA unavailable, and 0 devices; the main
  uv env has torch `2.12.0+cu130`, CUDA `13.0`, CUDA unavailable, and 0
  devices; `nvidia-smi` fails with GPU access blocked by the operating system.
  This keeps T002 blocked by host GPU/NVML access in this session rather than a
  pinned-env-only issue or a planner-stage API safety issue.
- 2026-05-25: Inspected `.pawl/version`, `.pawl/policy.yaml`, and
  `.pawl/migration-report.md`. They are expected schema metadata generated by
  the current pinned PawlKit run; `.pawl/policy.yaml` is declared as the policy
  source of truth and the migration report says legacy `AGENTS.md` was
  preserved.
- 2026-05-26: Marked T002 blocked on host GPU/NVML access while keeping it open
  for rerun after GPU visibility is restored. This blocker does not invalidate
  the existing runtime observer, advisory diagnostic, dry-run artifact, or the
  safety of future `get_kv_cache_configs(...)` observation.

## Verification Log

2026-05-25 planner-stage observation implementation:

- `PYTHONPATH=src /tmp/vllm-cachepawl-venv/bin/python benchmarks/scripts/capture_vllm_planner_stage_observation.py --output-dir research/avmp/v2/results/vllm-planner-stage-observation --timestamp 2026-05-25T00:00:00+00:00 --timeout-seconds 1200` — completed with structured blocker because torch reports CUDA unavailable in the pinned vLLM environment
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_vllm_planner_stage_observation.py -q` — 1 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 172 files already formatted
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py benchmarks/scripts/capture_vllm_cache_plan_observation.py benchmarks/scripts/capture_vllm_runtime_cache_plan_observation.py benchmarks/scripts/create_vllm_cache_diagnostic.py benchmarks/scripts/create_vllm_planner_dry_run_probe.py benchmarks/scripts/capture_vllm_planner_stage_observation.py` — passed, 170 source files
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 386 passed, 12 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build backend requirements
- `npx @codepawl/pawlkit@0.3.0 view` — passed after approved npm access; initial sandboxed attempt failed with registry DNS `EAI_AGAIN`
- `npx @codepawl/pawlkit@0.3.0 check` — passed with 0 warnings after approved npm access; initial sandboxed attempt failed with registry DNS `EAI_AGAIN`

2026-05-25 blocker stabilization:

- `/tmp/vllm-cachepawl-venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"` — `2.11.0+cu130 13.0 False 0`
- `/tmp/vllm-cachepawl-venv/bin/python -c "import vllm; print(getattr(vllm, '__version__', 'unknown'))"` — `0.21.0`
- `nvidia-smi --query-gpu=name,memory.free,memory.total,driver_version --format=csv,noheader` — failed: GPU access blocked by the operating system
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"` — `2.12.0+cu130 13.0 False 0`
