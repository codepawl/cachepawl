# RTX 3060 Planner Comparison

This artifact pack is a deterministic planner-only comparison for synthetic
hybrid KV/State cache workloads.

## Contents

- `short-heavy.jsonl`
- `long-heavy.jsonl`
- `mixed.jsonl`
- `summary.md`
- `environment.json`
- `manifest.json`

## Reproduction

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/create_planner_comparison_pack.py --output-dir benchmarks/results/rtx3060/planner-comparison --seed 1 --num-requests 128 --gpu-name "NVIDIA GeForce RTX 3060" --gpu-total-bytes 12884901888
```

Configuration:

- model: `jamba-1.5-mini`
- target GPU: `NVIDIA GeForce RTX 3060`
- target GPU bytes: `12884901888`
- seed: `1`
- requests per workload: `128`
- timestamp: `1970-01-01T00:00:00Z`
- runtime measurement: `false`

## Interpretation

- `vllm-style-padded` is a modeling baseline for uniform padded cache planning,
  not an exact measurement of vLLM internals.
- `cachepawl-avmp` is planner-only evidence, not runtime vLLM serving evidence.
- `overestimation_ratio` is `estimated_bytes / useful_bytes`.
- `wasted_fraction` is `(estimated_bytes - useful_bytes) / estimated_bytes`.
- `planner_runtime_us` is deterministic `0.000` unless runtime measurement is
  explicitly enabled.
- AVMP can reduce overestimation while `virtual_oom` may still be true when the
  useful demand itself exceeds the 12GB target profile.

No vLLM install, runtime serving, monkeypatching, allocator replacement, Triton
kernels, copy kernels, LSDR, or real model inference are used for this pack.
