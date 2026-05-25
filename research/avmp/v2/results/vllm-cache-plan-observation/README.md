# vLLM Cache Plan Observation

Status: `direct_real_object_translation`

This artifact is a read-only direct real vLLM object translation captured from
`vllm==0.21.0` in `/tmp/vllm-cachepawl-venv` with `PYTHONPATH=src`.
It constructs vLLM 0.21.0 cache planning dataclasses and translates them with
Cachepawl's import-safe translator. It does not load a model, call
`get_kv_cache_configs`, modify vLLM source, monkeypatch vLLM, replace
allocators, or implement Path C mutation.

## Files

- `manifest.json` — capture status, scope, and fake-vs-real comparison.
- `translated_cache_config.json` — Cachepawl-owned translated snapshot.
- `raw_safe_metadata.json` — signatures and scalar metadata only; no tensors.

## Fake-vs-Real Assumption Comparison

- `AttentionSpec.page_size_bytes`: matches — real vLLM exposes page_size_bytes as an observable property
- `AttentionSpec.dtype`: compatible — real vLLM uses torch.dtype; translator stringifies it
- `MambaSpec.shapes`: fake assumption widened — fake tests used dicts; real vLLM uses tuple[tuple[int, ...], ...]
- `MambaSpec.dtypes`: fake assumption widened — fake tests used dicts; real vLLM uses tuple[torch.dtype, ...]
- `KVCacheGroupSpec.layer_names`: compatible — real vLLM uses lists; translator normalizes to tuples
- `KVCacheConfig.block_size/cache_dtype`: not present — real vLLM 0.21.0 KVCacheConfig only has num_blocks, tensors, and groups

## Minimal Next Observe-First Step

Capture a runtime-resolved `KVCacheConfig` from a vanilla vLLM engine or worker
after planning, then run the same translator against that object.
