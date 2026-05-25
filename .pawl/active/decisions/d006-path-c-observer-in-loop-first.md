# D006 — Keep Path C observer-in-the-loop before mutation

Status: Accepted
Created: 2026-05-25
Updated: 2026-05-25
Completed: N/A
TTL: Keep while active
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Decision

Continue Path C with an observer-in-the-loop planner comparison and advisory
recommendation step before any scheduler, manager, allocator, or worker
allocation mutation.

## Reason

The runtime observer reached vanilla vLLM's resolved cache plan at
`LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config` and translated
it without mutating vLLM. The translated artifact contains enough stable fields
for Cachepawl-side planner metrics and recommendations:

- final `num_blocks`;
- available KV-cache GPU memory;
- cache group count, tensor count, and layer count;
- per-group cache kind, block size, padded page size, useful bytes, dtype, and
  layer names;
- Mamba state shapes, dtypes, mode, and padded page size;
- tensor sizes and shared layer names.

Those fields are enough to compare vLLM's observed reserved bytes against
Cachepawl's recommended native KV-page plus Mamba/state sizing. They are not
enough to safely replace vLLM allocation, change scheduler decisions, or alter
worker tensor layout because the observer sees the plan after vLLM has already
resolved scheduler and tensor allocation state.

## Consequences

- The next implementation should produce advisory planner-comparison records
  from the translated runtime observation.
- The advisory step must remain import-safe without vLLM installed.
- The capture path may log observer output around vanilla initialization, but it
  must not modify vLLM behavior.
- Scheduler/allocator mutation remains deferred until a stable control point is
  identified before finalized `KVCacheConfig` tensor sizes or inside a safe
  upstreamable construction hook.

## Alternatives Considered

- Mutate scheduler or `KVCacheManager` immediately.
- Replace worker tensor allocation layout immediately.
- Stop at static artifact documentation without building advisory comparison.

## Date

2026-05-25
