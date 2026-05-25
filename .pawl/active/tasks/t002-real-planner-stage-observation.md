# T002 — Real planner-stage observation around get_kv_cache_configs

Project: `.pawl/active/projects/project-main.md`
Sprint: `.pawl/active/sprints/sprint-002-planner-stage-observation.md`
Status: In Progress
Created: 2026-05-25
Updated: 2026-05-25
Completed: N/A
TTL: 30 days after completion or cancellation
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Objective

Determine whether Cachepawl can safely observe or call vLLM 0.21.0 planner-stage cache planning around `get_kv_cache_configs(...)` with real `VllmConfig`, `dict[str, KVCacheSpec]`, and available-memory inputs, without mutating vLLM behavior.

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

- [ ] The planner-stage observation attempt is bounded and reproducible
- [ ] The artifact directory `research/avmp/v2/results/vllm-planner-stage-observation/` contains `README.md` and `manifest.json`
- [ ] The artifact contains translated planner-stage output if real objects are reached
- [ ] The artifact contains `blocker.json` if safe planner-stage access is blocked
- [ ] The result compares planner-stage fields against the runtime observer and dry-run assumptions
- [ ] PawlKit validation is recorded
- [ ] Focused tests or checks are recorded if code is added or changed

## Verification

Use the commands in `.pawl/context/REPO_COMMANDS.md`. Product-code tests are required only if T002 adds or changes product code.

## Progress Notes

- 2026-05-25: Opened T002 after T001 closed the observe-first vLLM integration boundary. This task is limited to real planner-stage observation around `get_kv_cache_configs(...)` and must remain non-mutating.
- 2026-05-25: Accepted D008 to capture PawlKit issues only when normal
  Cachepawl work exposes real friction. No product code or research direction
  changed.
