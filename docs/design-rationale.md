# Design rationale

Why does a hybrid Mamba-Transformer-MoE workload need its own cache allocator?
Existing systems work fine for pure transformers; the answer below explains
where they break, and what Cachepawl is optimizing for as a result.

## Where existing cache solutions break

Production inference stacks (vLLM, SGLang, TensorRT-LLM) ship a KV cache
manager built around three implicit assumptions:

1. **Every layer has the same cache shape.** A KV block carries a fixed number
   of tokens; the per-token element count is the same on every layer. The
   block manager allocates the same physical block size for every layer and
   the per-layer index just multiplies.
2. **The cache grows append-only with the token count.** A sequence's cache
   footprint is `seq_len * per_token_bytes * num_layers`. Knowing `seq_len` is
   enough to know the cache footprint.
3. **Cross-sequence reuse follows the prompt tree.** Prefix caching exploits
   shared prompt prefixes; speculative decoding shares partial KVs across
   branches. The unit of reuse is "the first N tokens were the same."

Hybrid Mamba-Transformer-MoE models violate all three.

### Variable per-layer cache shapes

Attention layers and SSM layers carry structurally different state. An
attention layer at position `l` holds `seq_len` token-level KV vectors. An
SSM layer at position `l` holds a single fixed-size hidden state per sequence,
independent of `seq_len`. A fixed-block manager that allocates the same
physical block size for every layer wastes memory on SSM layers and
fragments under attention layers.

### Asymmetric reuse patterns

Prefix caching works on attention because attention KV is a function of the
token positions that produced it. SSM state is a recurrence: it is a function
of the prefix of tokens already consumed, but it is opaque to position. Two
sequences that share a prefix can share KV for that prefix, but their SSM
state has to be reconstructed from scratch unless you cache the state at the
end of the shared prefix specifically. The unit of reuse is different, and a
KV-only cache manager has no place to put the SSM state checkpoint.

### MoE-induced burstiness

MoE routing makes the per-token expert demand a runtime decision. Two
sequences in the same batch may need disjoint expert sets, which means the
demand on each expert's cache state and on the routing buffers is highly
variable across requests. A scheduler that assumes uniform per-sequence
cache footprint will mis-size the batch.

### The combined effect

In a model like Jamba or Zamba2, most layers are SSM-only and a minority are
attention; in Hymba, both run in parallel inside the same layer. A pure-KV
cache manager forced to handle these models ends up either
- allocating per-layer with the largest cache shape (massive waste), or
- forking the per-layer block size manually (fragile, hard to schedule).

Neither path scales to the model zoo. The cleanest fix is to model the two
cache kinds as first-class citizens with their own descriptors, their own
allocator surface, and a shared coordinator that owns scheduling decisions
across both kinds.

## What Cachepawl optimizes for

Cachepawl exists to make the following workloads first-class:

- **Hybrid inference at small-batch latency.** RTX 3060-class hardware, 8 to
  16 concurrent sequences, 8K to 64K context. The goal is not throughput
  records; it is making hybrid models actually fit and run.
- **Quantized cache types.** FP8 (E4M3, E5M2) and FP4 storage with packed
  sub-byte addressing. The dtype is part of the block descriptor from day
  one so the allocator can mix them without retrofitting.
- **Per-layer routing decisions.** The coordinator knows each layer's kind
  and routes reservation calls to the right manager. Adding a new layer kind
  is a model-spec change, not an allocator change.
- **Observable fragmentation and reuse.** The allocator stats surface
  fragmentation and per-kind occupancy; benchmarks can compare designs
  without instrumenting the kernels.

What Cachepawl explicitly does not optimize for in this phase:

- Multi-GPU sharding, NVLink-aware placement, or pipeline parallelism.
- Continuous batching policy. The allocator exposes the primitives; the
  scheduler is somebody else's job.
- Best-in-class throughput on pure transformers. vLLM and SGLang already
  serve that workload well.
