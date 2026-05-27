# vLLM Mamba Attention Contract Observation Summary

Status: `mamba_attention_contract_observation_with_field_blockers`

Request id: `0-a1ee70e3`

This artifact is a bounded, read-only Mamba/attention contract observation from
vanilla `vllm==0.21.0` in `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv` with
`PYTHONPATH=src`.

No tensors or large model objects were serialized. No vLLM source edits,
monkeypatching, allocator replacement, scheduler mutation, worker layout
mutation, returned Cachepawl plans, controlled substitution, Triton kernels,
copy kernels, LSDR, serving changes, or quality evaluation were performed.

## Fields

- `mamba_state_index_contract`: `blocked`
- `attention_block_table_view_contract`: `observed`
- `attention_metadata_builder_contract`: `observed`
- `mamba_state_tensor_contract`: `blocked`

## Field-Level Blockers

- `mamba_state_index_contract`: mamba_state_idx was not reachable with the live request id
- `mamba_state_tensor_contract`: Mamba state tensors were not safely reachable by stable runtime attributes
