# Sprint 1 — vLLM integration baseline and shim

Status: In Progress
Created: 2026-05-23
Updated: 2026-05-23
Completed: N/A
TTL: 30 days after completion or cancellation
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Goal

Prove cachepawl's Python AVMP allocator path inside vLLM by first capturing a vanilla baseline, then wiring the planned AVMP shim against the pinned vLLM version.

## Tasks

- [ ] `.pawl/active/tasks/t001-vllm-baseline-and-shim.md`

## Definition of Done

- [ ] vLLM development environment is documented and reproducible
- [ ] Vanilla vLLM baseline numbers are captured under `research/avmp/v2/results/`
- [ ] AVMP integration path is implemented or the fallback path is documented with evidence
- [x] New integration skeleton has focused tests and documented local verification
- [x] `ruff`, `ruff format --check`, `mypy`, and pytest status are recorded for the skeleton step

## Constraints

- Use `research/avmp/v2/VLLM_INTEGRATION_ROADMAP.md` as the controlling plan.
- Start with Path C: subclass `KVCacheManager` and inject via scheduler configuration.
- Pin `vllm==0.21.0` for reproducibility.
- Use Zamba2-2.7B-instruct as primary and Falcon-H1-1.5B-Instruct as backup.
- Keep `TritonAVMPAllocator` as correctness-oracle context only; production per-allocate Triton deployment remains deferred to v2.1.

## Non-Goals

- Do not refactor unrelated allocator or benchmark code.
- Do not change paper claims without updating the relevant research notes.
- Do not bypass failed checks by weakening tests, narrowing behavior, or deleting validation.

## Risks

- vLLM internals may not subclass cleanly; use the documented Path A fork fallback if Path C fails.
- WSL2 or 12 GiB GPU memory pressure may force the Falcon-H1 backup model.
- If vLLM upstream fixes the hybrid cache overestimation before evaluation, update the paper narrative before claiming results.

## Progress Notes

- 2026-05-23: Landed import-safe `cachepawl.integrations.vllm` skeleton and
  structural cache-plan translation records. Runtime vLLM serving, allocator
  replacement, and baseline metrics remain pending.
