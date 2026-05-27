# Limitations

This evaluation pack supports the advisory/diagnostic path only. It must not be
read as evidence that Cachepawl has performed controlled runtime substitution in
vLLM.

## What The Evidence Supports

- vLLM planner-stage replay can be observed and translated without changing the
  runtime scheduler config.
- The observed planner output matches the runtime scheduler cache config.
- The observed cache plan contains measurable reservation overestimation for
  this hybrid workload across four bounded config cells.
- `cachepawl diagnose-vllm` can report useful artifact-input advisory metrics
  without importing vLLM in the main Cachepawl environment.
- Request-to-block assignment, worker tensor layout, attention block-table/view
  metadata, and attention metadata builders are observable in bounded read-only
  runs.

## What The Evidence Does Not Support

- Runtime allocator replacement.
- Runtime cache substitution.
- Actual VRAM reduction during serving.
- Latency or throughput claims.
- Quality or accuracy claims.
- Generalization to every vLLM model, backend, or cache mode.
- Generalization beyond the single observed model,
  `Zyphra/Zamba2-2.7B-instruct`.
- Throughput, latency, quality, or actual serving-memory trends. The matrix is
  planner-stage advisory evidence only.

## Remaining Blockers

- `mamba_state_index_contract` remains blocked. The `mamba_state_idx` mapping
  was reachable, but it was empty for the live request.
- `mamba_state_tensor_contract` remains blocked. No Mamba state tensors were
  safely reachable by stable runtime attributes.
- The observed runtime cache config reported `mamba_cache_mode: none`, so this
  run did not populate the Mamba state-index/state-tensor paths needed for a
  rewrite contract.

Future mutation work requires either a run/model/config where Mamba state-index
and state tensors are observable, or a supported vLLM integration seam that
defines how an external allocator may provide or rewrite those views.
