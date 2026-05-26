# Sprint 3 — Path C mutation-hook design gate

Status: Completed
Created: 2026-05-26
Updated: 2026-05-26
Completed: 2026-05-26
TTL: 30 days after completion or cancellation
Archive After: 2026-06-25
Archive Warning: 2026-06-18
Archive Reason: Completed Path C mutation-hook design gate

## Goal

Use the successful T002 planner-stage replay evidence to choose the next bounded
Path C control point design before any mutation is implemented.

## Tasks

- [x] `.pawl/active/tasks/t005-mutation-hook-design-gate.md`

## Definition of Done

- [x] T005 compares candidate mutation-hook control points against the T002
  evidence
- [x] The comparison documents required control point, correctness risk,
  rollback strategy, advisory-mode verification, substitution-mode
  verification, and required pre-mutation tests
- [x] The sprint records a design recommendation or an explicit blocker for
  future mutation work
- [x] PawlKit validation is recorded

## Constraints

- Do not implement mutation in this sprint task.
- Do not modify vLLM source.
- Do not monkeypatch vLLM yet.
- Do not return any Cachepawl plan to vLLM.
- Do not replace allocators or alter scheduler behavior.
- Do not add vLLM to main Cachepawl dependencies.
- Do not add Triton kernels, copy kernels, LSDR, serving changes, or quality
  evaluation.

## Non-Goals

- Runtime mutation.
- Scheduler or allocator replacement.
- Worker tensor layout changes.
- Long-lived `vllm serve`.
- Model-quality evaluation.

## Progress Notes

- 2026-05-26: Opened Sprint 3 after Sprint 2 / T002 completed direct
  planner-stage replay. The sprint starts with T005, a design-only gate for
  comparing possible Path C mutation-hook control points.
- 2026-05-26: Completed Sprint 3. T005 selected planner-stage post-call
  advisory/diff as the next bounded experiment before any mutation.
- 2026-05-26: Final PawlKit validation passed with 0 warnings after approved
  network access; sandboxed npm access failed with DNS `EAI_AGAIN`.
