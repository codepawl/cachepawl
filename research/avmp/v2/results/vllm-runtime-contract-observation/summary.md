# vLLM Runtime Contract Observation Summary

Status: `runtime_contract_observation_with_field_blockers`

This artifact is a bounded, read-only runtime-contract observation from vanilla
`vllm==0.21.0` in `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv` with `PYTHONPATH=src`.

No vLLM source edits, monkeypatching, allocator replacement, scheduler
mutation, worker layout mutation, returned Cachepawl plans, controlled
substitution, Triton kernels, copy kernels, LSDR, serving changes, or quality
evaluation were performed.

## Fields

- `scheduler_kv_cache_manager_structure`: `observed`
- `block_usage_metadata`: `observed`
- `worker_cache_tensor_layout`: `observed`
- `request_to_block_assignment`: `blocked`
- `mamba_state_index_attention_view_contract`: `blocked`

## Field-Level Blockers

- `request_to_block_assignment`: get_block_ids exists but needs a live request id; no request was scheduled in this read-only capture
- `mamba_state_index_attention_view_contract`: Mamba state-index and attention block-table tensors were not safely reachable
