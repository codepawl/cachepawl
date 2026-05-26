# Sprint 2 — Planner-stage observation and diagnose-vllm CLI

Status: In Progress
Created: 2026-05-25
Updated: 2026-05-26
Completed: N/A
TTL: 30 days after completion or cancellation
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Goal

Observe real vLLM 0.21.0 planner-stage cache inputs and outputs around `get_kv_cache_configs(...)` without mutating vLLM behavior, then productize the existing observer/advisory diagnostic value while host GPU/NVML access blocks the planner-stage rerun.

## Tasks

- [x] `.pawl/active/tasks/t003-cachepawl-diagnose-vllm-cli.md`
- [ ] `.pawl/active/tasks/t002-real-planner-stage-observation.md` — blocked on host GPU/NVML access

## Definition of Done

- [ ] A bounded planner-stage observation attempt runs in `/tmp/vllm-cachepawl-venv` with `PYTHONPATH=src`
- [ ] The probe investigates real `VllmConfig`, `dict[str, KVCacheSpec]`, and available-memory inputs around `get_kv_cache_configs(...)`
- [ ] The result is recorded under `research/avmp/v2/results/vllm-planner-stage-observation/`
- [ ] Successful observations are translated through Cachepawl's import-safe vLLM translator
- [ ] Unsafe or unstable private/runtime access produces a structured blocker artifact instead of wider scope
- [ ] PawlKit validation and any focused code checks are recorded
- [x] The first user-facing diagnostic CLI mode can run from artifacts without vLLM, GPU, or NVML in the main environment

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
- 2026-05-25: Added the bounded planner-stage observation script and generated
  `research/avmp/v2/results/vllm-planner-stage-observation/`. The artifact is
  currently blocked by CUDA unavailability in `/tmp/vllm-cachepawl-venv`, so
  `get_kv_cache_configs(...)` was not called with real runtime inputs in this
  session. The artifact records safe static vLLM metadata and the real source
  call-site paths needed for the next GPU-visible rerun.
- 2026-05-25: Stabilized the T002 blocker diagnosis. CUDA is unavailable in
  both `/tmp/vllm-cachepawl-venv` and the main uv environment, and `nvidia-smi`
  reports host GPU access blocked by the operating system. The T002 artifact
  remains blocked and should be rerun unchanged after GPU/NVML visibility is
  restored.
- 2026-05-26: Kept T002 open but marked it blocked by host GPU/NVML access.
  Opened T003 for the next productization step: a `cachepawl diagnose-vllm`
  artifact-input CLI that reuses the existing translator, observer, advisory,
  dry-run, and runtime observation artifact format without requiring vLLM or GPU
  access in the main environment.
- 2026-05-26: Completed T003. Added `cachepawl diagnose-vllm` artifact-input
  mode, generated
  `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/`, and verified
  focused tests, full pytest, lint, format check, mypy, build, and PawlKit.
  Sprint 2 remains in progress only because T002 is blocked by host GPU/NVML
  access.
