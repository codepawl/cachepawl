# Path C Decision Gate

Date: 2026-05-25

Status: observer-in-the-loop first; scheduler or allocator mutation remains
deferred.

## Inputs

- Runtime artifact:
  `research/avmp/v2/results/vllm-runtime-cache-plan-observation/`
- Translated runtime config:
  `translated_runtime_cache_config.json`
- Raw runtime metadata:
  `raw_safe_metadata.json`
- Runtime path:
  `LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config`

The observation was produced from vanilla `vllm==0.21.0` in
`/tmp/vllm-cachepawl-venv` with no vLLM source edits, monkeypatching, allocator
replacement, scheduler injection, Path C mutation, long-lived serving, Triton
kernels, copy kernels, LSDR, or quality evaluation.

## Runtime Fields Available

- Final `KVCacheConfig.num_blocks`: `329`
- Runtime KV-cache memory metadata:
  - `available_gpu_memory_for_kv_cache`: `2915421184`
  - `cache_config_block_size`: `48`
  - `cache_config_num_gpu_blocks`: `329`
- Cache topology:
  - `group_count`: `7`
  - `attention_group_count`: `1`
  - `mamba_group_count`: `6`
  - `layer_count`: `63`
  - `tensor_count`: `9`
- Per-group fields:
  - cache kind: `attention` or `mamba`
  - spec type: `FullAttentionSpec` or `MambaSpec`
  - layer names
  - layer count
  - block size
  - padded page size bytes
  - useful bytes when derivable
  - dtype or per-state dtype metadata
- Attention metadata:
  - `num_kv_heads`: `32`
  - `head_size`: `160`
  - `dtype`: `torch.bfloat16`
  - `kv_quant_mode`: `0`
- Mamba metadata:
  - `shapes`: `[[3, 5248], [80, 64, 64]]`
  - `dtypes`: `[torch.bfloat16, torch.bfloat16]`
  - `mamba_cache_mode`: `none`
  - `mamba_type`: `MAMBA2`
  - `page_size_padded`: `983040`
  - `storage_block_size`: `4096`
- Tensor metadata:
  - nine `KVCacheTensor` records
  - each tensor size: `323420160` bytes
  - each tensor exposes `shared_by` layer names

## Sufficient For

### Planner-only comparison

The observer output is sufficient to produce runtime-grounded planner metrics:

- vLLM reserved bytes can be derived from `KVCacheTensor.size_bytes` records.
- Logical useful bytes can be derived from translated per-group useful bytes and
  `num_blocks`.
- Padding overhead can be estimated from padded page size versus useful bytes.
- Group and layer counts can be compared against Cachepawl's planner model.
- Runtime available KV memory can be used for virtual-OOM and capacity
  reporting.

This moves the comparison from synthetic model constants toward the actual
runtime-resolved vLLM plan.

### Observer-in-the-loop logging

The observer output is sufficient for a read-only helper that logs the resolved
vLLM plan during vanilla initialization:

- runtime path reached
- translated `KVCacheConfig`
- scalar memory metadata
- deterministic JSON payload
- unsupported-path reporting

This can be added around bounded vanilla captures without changing vLLM behavior.

### Future planner recommendation

The observer output is sufficient to compute Cachepawl-side recommendations such
as:

- estimated bytes under vLLM's padded plan
- estimated bytes under a Cachepawl native KV-page plus SSM-state plan
- estimated padding waste
- advisory capacity deltas
- per-group candidates where Mamba useful bytes are smaller than padded page
  bytes

These recommendations must remain advisory until a later mutation decision
identifies a safe vLLM control point.

## Insufficient For

### Replacing vLLM allocation

The translated runtime `KVCacheConfig` does not provide a sanctioned control
point for replacing allocation. It observes final tensor sizes and group
membership after planning, but does not expose a public allocator factory or a
stable way to replace the runtime `KVCacheManager`.

### Changing scheduler decisions

The observer sees `Scheduler.kv_cache_config` after scheduler construction. It
does not change scheduler admission, prefix-cache block tables, request block
assignment, or `KVCacheManager` behavior.

### Changing tensor allocation layout

The observer sees planned tensor sizes and shared layer names, but it does not
change `GPUModelRunner.initialize_kv_cache`, raw tensor allocation, strided
Mamba views, attention cache views, or block-table metadata. Layout changes
would require a future mutation point before or inside vLLM's worker allocation
path.

## Decision

Proceed with observer-in-the-loop as the next Path C step.

Do not implement scheduler or allocator mutation yet. The current observer
output has enough information to produce Cachepawl planner metrics and
recommendations without mutation. Mutation is only required when Cachepawl moves
from recommendations to enforcing a different allocation plan.

## Smallest Next Implementation Step

Add a Cachepawl-side runtime-plan comparison helper that consumes the translated
observer output and emits advisory records:

- vLLM observed reserved bytes
- vLLM observed useful bytes
- vLLM observed wasted fraction
- Cachepawl recommended bytes using native KV-page plus Mamba/state sizing
- advisory savings and capacity deltas
- unsupported/missing-field diagnostics

This helper should use existing artifact JSON or `VllmRuntimeCacheObservation`
objects and remain import-safe without vLLM installed.

## Deferred Mutation Gate

Before implementing mutation, capture or decide the exact control point for one
of these paths:

- planner-level change before `KVCacheConfig` tensor sizes are finalized;
- scheduler construction change that can inject a manager without monkeypatching;
- worker allocation change that can alter tensor layout while preserving Mamba
  state indices and attention backend expectations.

Until one of those control points is validated, Path C remains observe-first and
advisory.
