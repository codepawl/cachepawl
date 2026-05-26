# Sprint 5 — Mutation-readiness compatibility checks

Status: Completed
Created: 2026-05-26
Updated: 2026-05-26
Completed: 2026-05-26
TTL: 30 days after completion or cancellation
Archive After: 2026-06-25
Archive Warning: 2026-06-18
Archive Reason: Completed mutation-readiness compatibility checks

## Goal

Use the successful T002 planner-stage replay and T006 advisory/diff artifacts
to determine whether Cachepawl has enough serialized compatibility evidence
for a future controlled substitution experiment.

## Tasks

- [x] `.pawl/active/tasks/t007-mutation-readiness-compatibility-checks.md`

## Definition of Done

- [x] T007 consumes the T002 translated planner-stage config artifact
- [x] T007 consumes the T006 advisory diff and group-level diff artifacts
- [x] T007 emits a non-mutating readiness artifact under
  `research/avmp/v2/results/vllm-mutation-readiness/`
- [x] The artifact reports planner schema, num blocks, cache group count,
  cache tensor count, layer coverage, dtype/state dtype, block/page size,
  Mamba shape, attention/Mamba group mapping, estimated-byte consistency, and
  mutation-required missing-field checks
- [x] PawlKit validation is recorded

## Constraints

- Do not modify vLLM source.
- Do not monkeypatch vLLM.
- Do not return Cachepawl plans to vLLM.
- Do not replace allocators.
- Do not alter scheduler behavior or worker tensor layout.
- Do not require vLLM, GPU, or NVML.
- Do not add Triton kernels, copy kernels, LSDR, serving changes, or quality
  evaluation.

## Non-Goals

- Runtime mutation.
- Controlled return-value substitution.
- Scheduler or allocator replacement.
- Worker tensor layout changes.
- Long-lived `vllm serve`.
- Model-quality evaluation.

## Progress Notes

- 2026-05-26: Opened Sprint 5 for the pre-mutation safety gate selected after
  the T006 planner-stage advisory/diff artifact.
- 2026-05-26: Completed Sprint 5. T007 generated the mutation-readiness
  artifact and classified the current evidence as `advisory_only_recommended`
  because mutation-required control fields remain missing.
