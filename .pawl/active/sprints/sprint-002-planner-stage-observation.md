# Sprint 2 — Planner-stage vLLM observation

Status: In Progress
Created: 2026-05-25
Updated: 2026-05-25
Completed: N/A
TTL: 30 days after completion or cancellation
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Goal

Observe real vLLM 0.21.0 planner-stage cache inputs and outputs around `get_kv_cache_configs(...)` without mutating vLLM behavior, then record whether Cachepawl can compare against that stage directly or whether the private/runtime API boundary blocks the probe.

## Tasks

- [ ] `.pawl/active/tasks/t002-real-planner-stage-observation.md`

## Definition of Done

- [ ] A bounded planner-stage observation attempt runs in `/tmp/vllm-cachepawl-venv` with `PYTHONPATH=src`
- [ ] The probe investigates real `VllmConfig`, `dict[str, KVCacheSpec]`, and available-memory inputs around `get_kv_cache_configs(...)`
- [ ] The result is recorded under `research/avmp/v2/results/vllm-planner-stage-observation/`
- [ ] Successful observations are translated through Cachepawl's import-safe vLLM translator
- [ ] Unsafe or unstable private/runtime access produces a structured blocker artifact instead of wider scope
- [ ] PawlKit validation and any focused code checks are recorded

## Constraints

- Keep vLLM pinned at 0.21.0 in the isolated environment.
- Do not modify vLLM source.
- Do not monkeypatch vLLM.
- Do not return any Cachepawl plan to vLLM.
- Do not replace allocators or alter scheduler behavior.
- Do not add vLLM to main Cachepawl dependencies.
- Do not add Triton kernels, copy kernels, LSDR, serving changes, or quality evaluation.

## Non-Goals

- Runtime mutation.
- Scheduler or allocator replacement.
- Worker tensor layout changes.
- Long-lived `vllm serve`.
- Model-quality evaluation.

## Progress Notes

- 2026-05-25: Opened Sprint 2 after Sprint 1 closed the observe-first vLLM integration boundary. The next bounded step is real planner-stage observation around `get_kv_cache_configs(...)`.
- 2026-05-25: Accepted D008 as a lightweight PawlKit dogfooding policy. Record
  PawlKit feedback only when normal Cachepawl tracker work creates real
  friction; do not block Cachepawl research unless PawlKit validation is broken.
