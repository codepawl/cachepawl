# Architecture

This document describes the high-level architectural decision space for the
Cachepawl cache allocator and surveys prior art that motivates the design.
Implementation details are intentionally absent; concrete algorithms land in
follow-up design docs.

## The two-pool versus unified-pool tradeoff

A hybrid Mamba-Transformer-MoE model carries two structurally different cache
kinds in the same forward pass:

1. **KV cache blocks** for attention layers. Size scales with the per-sequence
   token count and the layer's head width. Block size is configurable but
   usually 16 or 32 tokens. Lifetime is "until the sequence ends or is evicted."
2. **SSM state blocks** for Mamba layers. Size is fixed per-layer regardless of
   sequence length, set by the SSM state dimension. Lifetime is "until the
   sequence ends." Typically one block per sequence per SSM layer.

The allocator has to fit both kinds into a single VRAM budget. There are two
clean ways to organize the underlying memory pool:

### Two-pool (asymmetric) design

Statically split the VRAM budget into a KV region and an SSM region at startup.
Each region is its own block pool with its own free list. The KV region uses
variable-count block reservation; the SSM region uses one-block-per-sequence
reservation.

- **Strengths:** simple bookkeeping, no cross-kind fragmentation, easy to tune
  per workload (long-context bias vs. high-batch bias).
- **Weaknesses:** the split is static. If a workload swings from
  attention-heavy to SSM-heavy mid-deployment, the underused region is dead
  memory. Cross-region rebalancing is not free.

### Unified-pool design

One block pool of fixed-size physical blocks. Each cache kind allocates from
the same pool and is responsible for packing its data into those blocks. SSM
state may need to span multiple physical blocks, and KV blocks must fit the
chosen physical block size cleanly.

- **Strengths:** no static split, automatic balance between kinds based on
  actual demand. Easier to evict across kinds.
- **Weaknesses:** packing logic is non-trivial. Internal fragmentation grows
  when block size does not match either cache kind. Eviction policy must rank
  blocks across two very different reuse patterns, which is hard to do well.

### Cachepawl's leaning

The asymmetric two-pool design is the natural starting point because it makes
the variable-versus-fixed shape distinction explicit and keeps the first
benchmark honest. A unified pool can be added once we have a workload trace
that proves the static split is costing more than the packing overhead would.

## Prior art

### vLLM `HybridKVCacheCoordinator`

vLLM introduced a coordinator that owns multiple `KVCacheGroup` objects, one
per layer kind. Each group has its own block manager and its own block layout.
The coordinator routes per-layer requests to the right group and exposes a
unified scheduler-facing API. This matches the "two pools, one coordinator"
shape and is the closest direct analogue to what Cachepawl needs.

### SGLang dual-pool design

SGLang separates the KV cache from request-level state caches and treats them
as cooperating pools. Block layout, eviction, and prefix caching are owned by
the KV pool; request state is owned by a thinner state pool. This split is
narrower than what hybrid models require (request state in SGLang is small and
opaque, not a per-layer SSM block), but the architectural pattern of distinct
cooperating pools is the same.

### Why not just reuse one of those

Neither system was designed for SSM state caches that scale with the number of
SSM layers and are reset on sequence end. Plugging an SSM cache into either
manager requires either a forked block type with custom shapes or a sidecar
allocator. Cachepawl exists to do the sidecar work once, cleanly, with a
shared eviction story.

## Open questions

These are deliberate gaps that the first concrete allocator design doc has to
answer:

- How does the eviction policy rank KV blocks against SSM blocks under
  pressure? Per-kind LRU is simple; cross-kind ranking needs a cost model.
- How does quantization (FP8, FP4) interact with block sizing? Sub-byte
  packing changes the effective block payload.
- How does MoE routing change the demand profile? Expert imbalance can spike
  one cache kind faster than the other.
- What is the right unit for sharing across sequences? Prefix sharing for KV
  is well understood; SSM state sharing is mostly unexplored.
