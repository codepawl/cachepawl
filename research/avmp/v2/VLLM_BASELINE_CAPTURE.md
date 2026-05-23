# Pinned vLLM Runtime Baseline Capture

This is the measurement-only path for Sprint 1 / T001. It captures the
vanilla vLLM baseline status before any Cachepawl allocator shim work.

## Pinned Environment

- vLLM: `vllm==0.21.0`
- Python: `3.10`
- isolated venv: `/tmp/vllm-cachepawl-venv`
- primary model: `Zyphra/Zamba2-2.7B-instruct`
- fallback model: `tiiuae/Falcon-H1-1.5B-Instruct`

Do not install vLLM into the Cachepawl repo venv and do not add vLLM to
`pyproject.toml`. Follow `research/avmp/v2/VLLM_DEV_SETUP.md` when the isolated
runtime environment is ready to be created.

## Baseline Status Capture

The first baseline capture is allowed to record a structured not-runnable result
when the current host cannot execute vanilla vLLM. The capture script records
vLLM availability, vLLM version if installed, torch/CUDA state, GPU metadata,
`nvidia-smi` status, target model config, and the blocker reason.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/capture_vllm_baseline.py \
  --output-dir research/avmp/v2/results/vllm-baseline \
  --model "Zyphra/Zamba2-2.7B-instruct" \
  --fallback-model "tiiuae/Falcon-H1-1.5B-Instruct" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 32 \
  --gpu-total-bytes 12884901888 \
  --gpu-name "NVIDIA GeForce RTX 3060"
```

Outputs:

- `research/avmp/v2/results/vllm-baseline/baseline.jsonl`
- `research/avmp/v2/results/vllm-baseline/manifest.json`
- `research/avmp/v2/results/vllm-baseline/README.md`

## Runtime Non-Goals

This step does not monkeypatch vLLM, replace allocators, add Triton kernels, add
copy kernels, add LSDR, or run real model quality evaluation. It records vanilla
runtime readiness only.
