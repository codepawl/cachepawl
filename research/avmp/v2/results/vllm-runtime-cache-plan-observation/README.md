# vLLM Runtime Cache Plan Observation

Status: `runtime_resolved_translation`

This artifact is a bounded, read-only runtime observation from vanilla
`vllm==0.21.0` in `/tmp/vllm-cachepawl-venv` with `PYTHONPATH=src`.
It initializes an offline `LLM` for `Zyphra/Zamba2-2.7B-instruct`, reads
`LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config`, and translates the resulting `KVCacheConfig` with
Cachepawl's import-safe translator.

No vLLM source edits, monkeypatching, allocator replacement, Path C mutation,
long-lived serving, Triton kernels, copy kernels, LSDR, or quality evaluation
were performed.

## Files

- `manifest.json` — capture status, path, parameters, and comparison.
- `translated_runtime_cache_config.json` — translated runtime planner output.
- `raw_safe_metadata.json` — scalar/list metadata only; no tensors or weights.

## Runtime vs Direct Observation

- `runtime_path`: new runtime-resolved object reached — scheduler.kv_cache_config is available after vanilla LLM initialization
- `KVCacheConfig.num_blocks`: runtime value — runtime planner resolved num_blocks=329
- `KVCacheConfig.kv_cache_groups`: runtime value — runtime planner resolved group_count=7
- `KVCacheConfig.kv_cache_tensors`: runtime value — runtime worker plan exposed tensor_count=9
- `direct dataclass observation`: translator assumptions still compatible — same translator handled real runtime KVCacheConfig without vLLM imports

## Minimal Next Observe-First Step

Convert this bounded script into an observer helper that can be invoked around
vanilla engine initialization and persist the translated config alongside
baseline runs.
