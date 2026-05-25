# D007 — Use a planner-level dry-run mutation probe next

Status: Accepted
Created: 2026-05-25
Updated: 2026-05-25
Completed: N/A
TTL: Keep while active
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Decision

Choose a planner-level dry-run probe as the next Path C mutation probe design.
Do not implement mutation yet.

The next probe should observe vLLM planner-stage inputs and outputs near
`get_kv_cache_configs(...)` or `get_kv_cache_config_from_groups(...)`, compute a
Cachepawl proposed alternate plan beside the vanilla plan, and persist both as
evidence without returning the alternate plan to vLLM.

## Reason

The advisory diagnostic showed meaningful planning waste:

- observed reserved bytes: 2,910,781,440
- advisory useful/recommended bytes: 1,679,258,112
- advisory savings bytes: 1,231,523,328
- overestimation ratio: 1.7333734577189286
- wasted fraction: 0.4230902777777778

To affect that waste, Cachepawl must eventually insert recommendations before
vLLM finalizes `KVCacheConfig` tensor sizes. The planner-level path is the
least invasive candidate that can test that control point. A dry-run version can
remain read-only and avoid vLLM source edits, monkeypatching, allocator
replacement, scheduler mutation, manager mutation, and worker allocation
mutation.

## Consequences

- The next implementation should be a bounded read-only planner-stage probe in
  the pinned vLLM environment.
- Scheduler construction and worker allocation hooks remain deferred.
- If planner-stage inputs cannot be reached without mutation, fall back to an
  observer-only recommendation CLI instead of escalating directly to scheduler
  or worker mutation.
- Actual runtime mutation requires a later decision after the dry-run probe
  proves input availability and contract compatibility.

## Alternatives Considered

- Scheduler construction hook: can affect runtime manager behavior, but is too
  late for finalized padded tensor sizes and has high private API risk because
  vLLM 0.21.0 directly constructs `KVCacheManager`.
- Worker allocation hook: can affect actual tensor layout, but has the highest
  correctness risk because it must preserve attention views, Mamba state
  indices, block tables, and backend metadata.
- Observer-only recommendation CLI: safest fallback if planner-stage inputs are
  inaccessible without mutation, but it cannot test insertion before vLLM
  finalizes planning.

## Date

2026-05-25
