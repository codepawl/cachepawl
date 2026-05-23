# Main Project

Status: Active
Created: 2026-05-23
Updated: 2026-05-23
Completed: N/A
TTL: 30 days after completion or cancellation
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Purpose

Cachepawl is a hybrid KV and SSM cache allocator for Mamba-Transformer-MoE language model inference. It owns a shared VRAM budget across attention KV pages and SSM state blocks, with Python AVMP prototypes, benchmark tooling, Triton correctness-oracle work, and research/paper artifacts.

## Current Sprint

`.pawl/active/sprints/sprint-001-vllm-integration.md`

## Current Task

`.pawl/active/tasks/t001-vllm-baseline-and-shim.md`

## Active Constraints

- Follow `.pawl/context/PRODUCT_SCOPE.md`.
- Follow `.pawl/context/TECHNICAL_SCOPE.md`.
- Keep project, sprint, task, and decision bodies in separate files.

## Notes

- v1 Python AVMP prototype is published as arXiv:2605.22416.
- v2 Triton hardware realization is a correctness oracle; production batched/deferred deployment is v2.1.
- The current implementation milestone is vLLM integration for the ML for Systems @ NeurIPS 2026 workshop path.
