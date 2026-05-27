# D010 — Control-field go/no-go for substitution

Status: Accepted
Created: 2026-05-26
Updated: 2026-05-27
Completed: 2026-05-27
TTL: Keep while active
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Decision

Stay advisory-only for the current cycle.

T008 produced `research/avmp/v2/PATH_C_CONTROL_FIELD_GATE.md`. The gate
classifies the scheduler/planner construction hook and allocator/KVCacheManager
replacement control point as `hard_blocker_for_current_cycle`, and classifies
the worker tensor layout, runtime request-to-block assignment, and Mamba
state-index/attention view rewrite contracts as
`resolvable_by_read_only_observation`.

The next task must not be a controlled substitution experiment. It should be a
bounded read-only runtime contract observation task if the project continues
Path C this cycle.

## Required Evidence

T008 must classify each unresolved mutation-control field as one of:

- `resolved_by_existing_evidence`
- `resolvable_by_read_only_observation`
- `requires_default_off_mutation_probe`
- `hard_blocker_for_current_cycle`

Fields:

1. stable scheduler or planner construction hook
2. allocator or KVCacheManager replacement control point
3. worker tensor allocation layout control point
4. runtime request-to-block assignment control
5. Mamba state-index and attention view rewrite contract

## Allowed Outcomes

- Authorize T009 as a default-off controlled substitution experiment only if
  the required control fields are resolved or intentionally gated with clear
  rollback and verification boundaries.
- Keep the project advisory-only for this cycle if any field remains a hard
  blocker or would require unsafe private-runtime mutation.

## Result

Selected outcome: keep the project advisory-only for this cycle.

Do not implement substitution, return modified planner results to vLLM, modify
vLLM source, monkeypatch, replace allocators, alter scheduler behavior, alter
worker tensor layout, or change Mamba/attention runtime metadata in T009.

## Next Task Shape

If opened, T009 should be read-only runtime contract observation:

- observe `scheduler.kv_cache_manager` around one bounded vanilla request
- record usage/free-block counts and request block ids when safely observable
- record worker tensor allocation/view shapes and strides when safely
  observable
- record Mamba block-table and state-index tensor shapes when safely observable
- stop if observation requires monkeypatching, object replacement, scheduler
  mutation, or worker layout mutation

## Constraints

This proposed decision does not authorize substitution, vLLM source edits,
monkeypatching, allocator replacement, scheduler mutation, worker layout
mutation, Triton kernels, copy kernels, LSDR, serving changes, or quality
evaluation.

## Date

2026-05-26

## Resolution Date

2026-05-27
