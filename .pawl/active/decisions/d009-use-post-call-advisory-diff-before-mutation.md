# D009 — Use post-call advisory diff before mutation

Status: Accepted
Created: 2026-05-26
Updated: 2026-05-26
Completed: N/A
TTL: Keep while active
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Decision

Use a planner-stage post-call advisory/diff artifact as the next bounded Path C
experiment. Do not implement mutation yet.

The next experiment should run after vanilla
`vllm.v1.core.kv_cache_utils.get_kv_cache_configs(...)` returns, compute a
Cachepawl proposed planner result beside the vanilla result, and persist a
structured diff without returning the Cachepawl proposal to vLLM.

## Reason

T002 proved the real planner-stage boundary is reachable with the durable
pinned vLLM environment:

- real `vllm_config`, `kv_cache_specs`, and `available_memory` were reached
- one worker and 63 layer/spec entries were observed
- spec types were `FullAttentionSpec=9` and `MambaSpec=54`
- available memory was `2915421184`
- vanilla planner and runtime scheduler both used `num_blocks=329`
- `planner_matches_runtime_scheduler=true`
- `runtime_changed_during_replay=false`

The post-call advisory/diff path is the smallest next experiment that can
compare vanilla and Cachepawl planner results while keeping vLLM behavior
unchanged. Controlled return-value substitution remains deferred until the diff
artifact proves the required compatibility and rollback gates.

## Consequences

- Normal CLI and advisory modes remain non-mutating.
- Any future mutation requires a default-off feature flag, explicit opt-in,
  structured before/after artifact, rollback path, and vanilla parity check.
- Pre-call wrapping, controlled substitution, and Scheduler/EngineCore hooks
  remain deferred until post-call diff evidence justifies them.
- No vLLM source edits, monkeypatching, allocator replacement, returned
  Cachepawl plans, scheduler mutation, worker layout mutation, Triton kernels,
  copy kernels, LSDR, serving changes, or quality evaluation are authorized by
  this decision.

## Alternatives Considered

- Pre-call wrapper around `get_kv_cache_configs(...)`: earlier, but likely
  requires monkeypatching or a fork and has higher input-mutation risk.
- Controlled isolated return-value substitution: plausible future mutation
  path, but too risky before structural compatibility and rollback evidence
  exist.
- Scheduler/EngineCore hook: higher private API and correctness risk because it
  touches scheduler, manager, and block-accounting state.

## Date

2026-05-26
