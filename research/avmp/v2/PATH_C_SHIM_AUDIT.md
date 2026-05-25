# Path C Shim Audit for vLLM 0.21.0

Date: 2026-05-23
Scope: read-only audit of `/tmp/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm`
Baseline boundary: `research/avmp/v2/results/vllm-baseline/`

## Status

The pinned vanilla baseline is complete enough to begin Path C design:

- `vllm==0.21.0` imports in `/tmp/vllm-cachepawl-venv`.
- CUDA is visible on the local WSL2 RTX 3060 path.
- `Zyphra/Zamba2-2.7B-instruct` loads with vanilla vLLM.
- A bounded one-prompt generation smoke completed.

This audit did not modify vLLM source, monkeypatch vLLM, replace allocators,
serve a long-running endpoint, add kernels, add LSDR, or run quality evaluation.

## Files Audited

- `/tmp/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm/v1/kv_cache_interface.py`
- `/tmp/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm/v1/core/kv_cache_utils.py`
- `/tmp/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm/v1/core/kv_cache_manager.py`
- `/tmp/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm/v1/core/kv_cache_coordinator.py`
- `/tmp/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm/v1/core/single_type_kv_cache_manager.py`
- `/tmp/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py`
- `/tmp/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py`
- `/tmp/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm/v1/worker/utils.py`
- `/tmp/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm/v1/attention/backends/mamba_attn.py`
- `/tmp/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm/config/cache.py`
- `/tmp/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm/config/vllm.py`

## Candidate vLLM Objects

### Cache spec model

`vllm.v1.kv_cache_interface` defines the main structural metadata:

- `KVCacheSpec`: base per-layer cache format with `block_size`,
  `page_size_bytes`, `storage_block_size`, `max_memory_usage_bytes`, and
  `copy_with_new_block_size`.
- `AttentionSpec`: attention cache format with `num_kv_heads`, `head_size`,
  `dtype`, `kv_quant_mode`, optional `page_size_padded`, and
  `real_page_size_bytes`.
- `MambaSpec`: Mamba/state cache format with `shapes`, `dtypes`, optional
  `page_size_padded`, `mamba_type`, `mamba_cache_mode`, and
  `num_speculative_blocks`.
- `UniformTypeKVCacheSpecs`: wrapper for layers with the same cache behavior
  but potentially different `page_size_bytes`.
- `KVCacheGroupSpec`: layer group plus merged `kv_cache_spec`.
- `KVCacheConfig`: final worker/scheduler cache plan with `num_blocks`,
  `kv_cache_tensors`, and `kv_cache_groups`.

These are the best read-only metadata surface for Cachepawl. They expose the
same concepts the planner-comparison work already models: per-layer page size,
block size, group membership, useful versus padded page size, and Mamba cache
mode.

### Planner and grouping path

`vllm.v1.core.kv_cache_utils` is the narrowest planning path:

- `get_kv_cache_configs(...)` merges worker specs, derives global groups,
  projects groups back to workers, checks memory, and returns per-worker
  `KVCacheConfig` values.
- `get_kv_cache_groups(...)` chooses between uniform spec, uniform type,
  Deepseek-specific grouped uniform specs, and general hybrid grouping.
- `unify_kv_cache_spec_page_size(...)` raises smaller page sizes to the maximum
  page size by increasing `block_size` when possible.
- `_get_kv_cache_groups_uniform_page_size(...)` documents vLLM's current
  hybrid assumptions: equal physical memory per block across groups, shared
  group size, and same page size after unification.
- `get_kv_cache_config_from_groups(...)` converts groups and available memory
  into `num_blocks` and `KVCacheTensor` allocations.
- `_report_kv_cache_config(...)` logs GPU KV cache size and max concurrency,
  matching the baseline smoke output.

For read-only observation, this function family is enough to reconstruct the
planner decision after vLLM has already produced it. For future mutation, this
is a tempting but high-risk hook because it is a module-level internal function,
not a documented extension point.

### Runtime cache manager path

`vllm.v1.core.sched.scheduler.Scheduler.__init__` directly constructs
`KVCacheManager(...)` and stores it at `self.kv_cache_manager`.

`vllm.v1.core.kv_cache_manager.KVCacheManager` delegates most behavior to a
coordinator from `get_kv_cache_coordinator(...)`. Public-ish methods that a
future shim would need to preserve are:

- `get_computed_blocks`
- `allocate_slots`
- `free`
- `remove_skipped_blocks`
- `evict_blocks`
- `reset_prefix_cache`
- `get_num_common_prefix_blocks`
- `take_events`
- `get_blocks`
- `get_block_ids`
- `cache_blocks`
- `take_new_block_ids`
- `new_step_starts`

`vllm.v1.core.kv_cache_coordinator` then fans out to per-type managers. The
hybrid case uses `HybridKVCacheCoordinator`, which assumes group block sizes are
divisible by `hash_block_size`, rejects DCP/PCP, and computes prefix hits across
attention groups.

`vllm.v1.core.single_type_kv_cache_manager.MambaManager` is the key Mamba state
allocation behavior. In `mamba_cache_mode == "align"`, it allocates at most one
new running-state block per step after the first allocation, tracks
`last_state_block_idx`, and reuses previous speculative blocks.

### Worker allocation path

`vllm.v1.worker.gpu_model_runner.GPUModelRunner` receives the resolved
`KVCacheConfig` and performs tensor allocation:

- `get_kv_cache_spec()` extracts per-layer `KVCacheSpec` from attention modules.
- `initialize_kv_cache(...)` deep-copies `KVCacheConfig`, initializes attention
  backends, prepares kernel block sizes, builds metadata builders, initializes
  tensors, and binds caches into static forward context.
- `_allocate_kv_cache_tensors(...)` allocates raw int8 tensors according to
  `KVCacheTensor.size` and `shared_by`.
- `_reshape_kv_cache_tensors(...)` creates attention and Mamba tensor views.
  It handles `page_size_padded` through strided views.
- `_update_hybrid_attention_mamba_layout(...)` changes attention layout when
  attention and Mamba caches coexist.

This is the best place to observe actual allocation layout after planning, but
it is too late to reduce planner overestimation without replacing upstream plan
inputs.

### Mamba metadata path

`vllm.v1.attention.backends.mamba_attn` builds runtime metadata from block
tables:

- `BaseMambaAttentionMetadataBuilder.__init__` allocates decode metadata buffers
  based on `MambaSpec.block_size` and `mamba_cache_mode`.
- `_compute_common_metadata(...)` derives `state_indices_tensor` through
  `mamba_get_block_table_tensor(...)` unless `mamba_cache_mode == "all"`.
- `_compute_prefix_caching_block_indices(...)` derives Mamba prefix-caching
  block indices.
- `update_block_table(...)` can refresh Mamba state indices for a changed block
  table.

This path is useful for validating that future AVMP block-table changes still
feed Mamba kernels correctly, but it is not the right first mutation point.

## Metadata Cachepawl Needs

From vLLM planning and runtime objects:

- vLLM version and selected model.
- `cache_config.block_size`, `cache_config.hash_block_size`,
  `cache_config.mamba_block_size`, `cache_config.mamba_cache_mode`,
  `cache_config.mamba_page_size_padded`, `cache_config.cache_dtype`,
  `cache_config.gpu_memory_utilization`, and `cache_config.kv_cache_memory_bytes`.
- Per-layer `KVCacheSpec` class name.
- Per-layer `block_size`.
- Per-layer `page_size_bytes`.
- For `AttentionSpec`: `real_page_size_bytes`, `num_kv_heads`, `head_size`,
  `dtype`, `kv_quant_mode`, and `page_size_padded`.
- For `MambaSpec`: `shapes`, `dtypes`, `mamba_type`, `mamba_cache_mode`,
  `num_speculative_blocks`, and `page_size_padded`.
- Final `KVCacheConfig.num_blocks`.
- Final `KVCacheConfig.kv_cache_groups`: group id, layer names, spec type,
  block size, page size, and `is_eagle_group`.
- Final `KVCacheConfig.kv_cache_tensors`: tensor size and shared layer names.
- Runtime allocator state, if observing a live manager: free block count,
  total block count, per-request block ids, and per-group block ids.

## Observable Without Mutation

The next implementation can stay read-only by adding a Cachepawl-side translator
that accepts vLLM objects and emits a structured snapshot:

- `GPUModelRunner.get_kv_cache_spec()` output can be translated if a runner is
  available.
- `KVCacheConfig` can be translated after vLLM planning but before any mutation.
- `Scheduler.kv_cache_manager.kv_cache_config` can be inspected after scheduler
  construction.
- `KVCacheManager.block_pool` exposes aggregate block usage through existing
  methods like `get_usage()` and free-block count through the pool object.
- `KVCacheManager.get_block_ids(request_id)` can observe allocated blocks for a
  request without modifying them.

The safest first probe is a pure translator for `KVCacheConfig` and
`dict[str, KVCacheSpec]`. It can run in the main Cachepawl environment against
test doubles and in the pinned vLLM env with `PYTHONPATH=src` when vLLM exists.

## Future Shim Attachment Points

### Recommended Path C sequence

1. Add import-safe Cachepawl translators for vLLM `KVCacheSpec`,
   `KVCacheGroupSpec`, `KVCacheTensor`, and `KVCacheConfig`.
2. Add a read-only runtime capture mode that records the translated
   `KVCacheConfig` from a vanilla vLLM engine.
3. Add a narrow `CachepawlKVCacheManager` subclass or wrapper only after the
   translated baseline proves which manager/coordinator calls must change.
4. Investigate scheduler construction injection. In vLLM 0.21.0,
   `Scheduler.__init__` constructs `KVCacheManager` directly, so a clean
   no-monkeypatch attachment point is not obvious from the installed package.

### Candidate hooks

- Scheduler construction hook:
  Replace or parameterize the direct `KVCacheManager(...)` construction in
  `Scheduler.__init__`. This matches the original Path C intent, but vLLM
  0.21.0 does not appear to expose a public factory for this.
- Planner hook:
  Wrap `get_kv_cache_configs(...)` or `get_kv_cache_config_from_groups(...)` to
  emit or alter `KVCacheConfig`. This is close to the overestimation source but
  would rely on module-private functions.
- Runtime manager hook:
  Subclass or wrap `KVCacheManager` while preserving all current public methods
  and the coordinator contract. This is plausible for allocation behavior, but
  it may be too late to fix page-size padding already baked into `KVCacheConfig`.
- Worker observation hook:
  Observe `GPUModelRunner.initialize_kv_cache(...)` and
  `_reshape_kv_cache_tensors(...)` outputs. This is low risk for telemetry but
  not sufficient for memory savings.

## Risks

- vLLM 0.21.0 cache planning APIs are internal and may change without
  compatibility guarantees.
- `Scheduler.__init__` directly constructs `KVCacheManager`, so Path C likely
  needs either an upstreamable factory seam, a local fork, or a carefully scoped
  monkeypatch in a later task.
- Page-size padding is resolved before tensor allocation. A manager-only shim may
  observe but not reduce `KVCacheTensor.size` unless it also changes the
  `KVCacheConfig`.
- `MambaManager` has mode-specific behavior, especially `align` mode, that must
  be preserved exactly.
- `HybridKVCacheCoordinator` encodes prefix-cache assumptions around
  `hash_block_size`, block-size LCM, and DCP/PCP restrictions.
- `GPUModelRunner._reshape_kv_cache_tensors(...)` relies on `page_size_padded`
  and strided views; any future AVMP layout must remain compatible with Mamba
  state indices and attention backend block dimensions.

## Minimal Next Implementation Step

Add an import-safe Cachepawl translator module under
`src/cachepawl/integrations/vllm/` that can convert vLLM 0.21.0 cache objects
into Cachepawl-owned typed records:

- `translate_kv_cache_spec(layer_name, spec)`
- `translate_kv_cache_group(group_id, group)`
- `translate_kv_cache_tensor(tensor)`
- `translate_kv_cache_config(config)`

The translator should use duck typing or optional imports so importing
Cachepawl remains safe without vLLM installed. Tests should use lightweight
fake objects in the main environment and, optionally, a pinned-env smoke command
to confirm real-object compatibility later.

## Translator Limitations

The first Cachepawl-side translator is intentionally observe-first. It accepts
duck-typed cache planning objects and emits serializable snapshots for
`AttentionSpec`, `MambaSpec`, `KVCacheGroupSpec`, `KVCacheTensor`, and
`KVCacheConfig`-like objects, but it does not import vLLM, monkeypatch vLLM,
replace allocators, or change scheduler behavior. Unsupported objects raise a
typed Cachepawl translation error instead of falling through to raw attribute
errors. Real vLLM object compatibility still needs a pinned-env smoke check
after a vanilla runtime capture exposes the resolved `KVCacheConfig`.

Do not change vLLM behavior in that next step. The goal is to capture the
runtime planning contract before replacing any allocator or scheduler behavior.
