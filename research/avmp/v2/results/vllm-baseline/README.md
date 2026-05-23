# vLLM Runtime Baseline Capture

This directory records the measurement-only vanilla vLLM baseline status for
Sprint 1 / T001.

## Contents

- `baseline.jsonl`
- `manifest.json`
- `README.md`

## Reproduction

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/capture_vllm_baseline.py --output-dir research/avmp/v2/results/vllm-baseline --model "Zyphra/Zamba2-2.7B-instruct" --fallback-model "tiiuae/Falcon-H1-1.5B-Instruct" --timestamp "1970-01-01T00:00:00Z" --max-model-len 4096 --gpu-memory-utilization 0.9 --max-num-seqs 32 --gpu-total-bytes 12884901888 --gpu-name "NVIDIA GeForce RTX 3060"
```

## Status

- status: `not_runnable`
- reason: `vllm is not installed in the active Python environment`
- primary model: `Zyphra/Zamba2-2.7B-instruct`
- fallback model: `tiiuae/Falcon-H1-1.5B-Instruct`
- pinned vLLM: `0.21.0`
- isolated venv: `/tmp/vllm-cachepawl-venv`

This step does not install vLLM, serve a model, monkeypatch vLLM, replace
allocators, add Triton kernels, add copy kernels, add LSDR, or run real model
quality evaluation.
