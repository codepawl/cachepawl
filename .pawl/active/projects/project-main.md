# Main Project

Status: Active
Created: 2026-05-23
Updated: 2026-05-26
Completed: N/A
TTL: 30 days after completion or cancellation
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Purpose

Cachepawl is a hybrid KV and SSM cache allocator for Mamba-Transformer-MoE language model inference. It owns a shared VRAM budget across attention KV pages and SSM state blocks, with Python AVMP prototypes, benchmark tooling, Triton correctness-oracle work, and research/paper artifacts.

## Current Sprint

None. Sprint 4 is complete.

## Current Task

None. T006 is complete.

## Active Constraints

- Follow `.pawl/context/PRODUCT_SCOPE.md`.
- Follow `.pawl/context/TECHNICAL_SCOPE.md`.
- Keep project, sprint, task, and decision bodies in separate files.

## Notes

- v1 Python AVMP prototype is published as arXiv:2605.22416.
- v2 Triton hardware realization is a correctness oracle; production batched/deferred deployment is v2.1.
- The current implementation milestone is vLLM integration for the ML for Systems @ NeurIPS 2026 workshop path.
- T002 completed after direct planner-stage replay through the durable pinned
  vLLM env. The artifact reached real planner inputs, called
  `get_kv_cache_configs(...)`, and confirmed the translated planner output
  matches the runtime scheduler config without vLLM mutation.
- T003 completed the GPU-free artifact-input `cachepawl diagnose-vllm` CLI.
- T004 completed the release-readiness README documentation follow-up for the diagnostic CLI.
- Sprint 3 / T005 opened as a design-only gate for comparing bounded Path C
  mutation-hook control points before any mutation was implemented.
- T005 completed the design gate and D009 selected planner-stage post-call
  advisory/diff as the next bounded experiment before mutation.
- Sprint 4 / T006 opened to produce a non-mutating planner-stage advisory/diff
  artifact from the T002 translated planner output.
- T006 completed the planner-stage post-call advisory/diff artifact under
  `research/avmp/v2/results/vllm-planner-stage-advisory-diff/`.
