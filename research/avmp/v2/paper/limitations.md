# Paper Limitations

This paper skeleton is intentionally scoped to a diagnostic/advisory systems
artifact.

## No Runtime Mutation Claim

The evaluation does not replace vLLM allocators, rewrite cache views, alter
scheduler behavior, alter worker tensor layout, or return Cachepawl plans to
vLLM. The reported savings are estimated advisory savings from planner artifacts,
not measured serving-time memory savings.

## No Performance Or Quality Claim

The evidence does not include:

- runtime VRAM reduction measurements,
- throughput measurements,
- latency measurements,
- serving experiments,
- model quality or accuracy evaluation.

## Single Observed Configuration

The current evidence is from one model and configuration:

- `Zyphra/Zamba2-2.7B-instruct`
- vanilla `vllm==0.21.0`
- `max_model_len=4096`
- `max_num_seqs=1`
- `gpu_memory_utilization=0.7`

The result should not be generalized to all vLLM versions, models, backends,
cache modes, or workloads without further evaluation.

## Mamba State Contract Blocker

The remaining blocker is Mamba state observability. The Mamba/attention
contract observation resolved attention block-table/view metadata and attention
metadata builders, but did not resolve:

- `mamba_state_index_contract`, because `mamba_state_idx` was reachable but
  empty for the live request;
- `mamba_state_tensor_contract`, because no Mamba state tensors were safely
  reachable by stable runtime attributes.

The observed runtime cache config reported `mamba_cache_mode: none`, so this run
did not populate the Mamba state-index/state-tensor paths needed for a rewrite
contract.

## Future Work

Future work must first improve Mamba state-index observability. A controlled
substitution experiment should only be considered after one of the following is
available:

- a run/model/config where Mamba state-index and state tensors are observable
  through bounded read-only runtime paths;
- a supported vLLM integration seam for external cache allocators or view
  rewrites;
- a default-off mutation probe with explicit rollback controls and validated
  Mamba/attention rewrite contracts.

The next evaluation step is multi-model and multi-workload advisory evaluation,
still without conflating advisory savings with measured runtime improvements.
