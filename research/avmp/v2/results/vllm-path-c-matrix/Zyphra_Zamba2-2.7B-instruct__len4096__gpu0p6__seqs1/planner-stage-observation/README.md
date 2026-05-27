# vLLM Planner-Stage Observation

Status: `planner_stage_translation`

This artifact records a bounded, read-only planner-stage observation from
vanilla `vllm==0.21.0` in `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv` with
`PYTHONPATH=src`. It reaches `vllm.v1.core.kv_cache_utils.get_kv_cache_configs` by replaying the
planner directly on real inputs observed after vanilla `LLM` initialization.

The computed Cachepawl translation is not returned to vLLM. No vLLM source
edits, monkeypatching, allocator replacement, scheduler mutation, worker layout
mutation, long-lived serving, Triton kernels, copy kernels, LSDR, or quality
evaluation were performed.

## Files

- `manifest.json` — capture status, parameters, non-mutation flags, and paths.
- `translated_planner_stage_config.json` — translated planner-stage output.
- `raw_safe_metadata.json` — scalar/list metadata only; no tensors or weights.

## Real Input Availability

- `VllmConfig`: available from `LLM.llm_engine.engine_core.engine_core.vllm_config`.
- `KVCacheSpec` maps: available from `LLM.llm_engine.engine_core.engine_core.model_executor.get_kv_cache_specs()`.
- Available memory: available from `LLM.llm_engine.engine_core.engine_core.available_gpu_memory_for_kv_cache`.

## Result

- `observation_mode`: `post_init_direct_replay_on_real_inputs`
- `kv_cache_specs_worker_count`: 1
- `kv_cache_specs_layer_counts`: [63]
- `available_memory`: [1626983424]
- `planner_output_num_blocks`: [183]
- `runtime_num_blocks`: 183
- `runtime_num_blocks_after_replay`: 183
- `runtime_changed_during_replay`: False
- `planner_matches_runtime_scheduler`: True

## Minimal Next Step

Use this same direct planner-stage observation as the source for a dry-run
comparison against Cachepawl recommendations. Runtime behavior remains
unchanged until a later explicit mutation decision identifies a safe scheduler
or allocator control point.
