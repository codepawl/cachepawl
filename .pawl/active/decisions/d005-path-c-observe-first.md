# D005 — Observe vLLM cache plans before Path C mutation

Status: Accepted
Created: 2026-05-23
Updated: 2026-05-23
Completed: N/A
TTL: Keep while active
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Decision

Start Path C with a read-only Cachepawl translator for vLLM cache planning
objects before implementing a `KVCacheManager` subclass, monkeypatch, allocator
replacement, or local fork.

## Reason

The vLLM 0.21.0 audit found multiple plausible integration paths:

- scheduler construction, where `Scheduler.__init__` directly creates
  `KVCacheManager`;
- planner utilities, where `get_kv_cache_configs` and related helpers build
  `KVCacheConfig`;
- runtime cache manager/coordinator classes, where allocation behavior is
  dispatched;
- worker allocation, where the resolved `KVCacheConfig` becomes real tensors.

The narrowest low-risk next step is to translate and record vLLM's resolved
`KVCacheSpec` and `KVCacheConfig` objects without mutation. That preserves the
completed vanilla baseline boundary and provides evidence for the later shim
choice.

## Consequences

- The next product-code step should add import-safe translators under
  `src/cachepawl/integrations/vllm/`.
- The translator must work without vLLM installed by using duck typing or
  optional imports.
- A future mutation step may still choose scheduler injection, a local fork, or a
  scoped monkeypatch, but that choice should be based on translated runtime
  evidence.
- Do not replace allocators, monkeypatch vLLM, implement Path C behavior, add
  kernels, add LSDR, or run quality evaluation as part of the translator step.

## Alternatives Considered

- Directly subclass and inject `KVCacheManager` through scheduler construction.
- Wrap or monkeypatch `get_kv_cache_configs`.
- Modify worker tensor allocation in `GPUModelRunner`.
- Fork vLLM immediately.

## Date

2026-05-23
