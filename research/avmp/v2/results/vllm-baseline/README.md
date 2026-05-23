# vLLM Runtime Baseline Capture

This directory records the measurement-only vanilla vLLM baseline status for
Sprint 1 / T001.

## Contents

- `baseline.jsonl`
- `manifest.json`
- `README.md`

## Reproduction

```bash
PYTHONPATH=src /tmp/vllm-cachepawl-venv/bin/python benchmarks/scripts/capture_vllm_baseline.py --output-dir research/avmp/v2/results/vllm-baseline --model "Zyphra/Zamba2-2.7B-instruct" --fallback-model "tiiuae/Falcon-H1-1.5B-Instruct" --timestamp "1970-01-01T00:00:00Z" --max-model-len 4096 --gpu-memory-utilization 0.7 --max-num-seqs 1 --gpu-total-bytes 12884901888 --gpu-name "NVIDIA GeForce RTX 3060" --runtime-timeout-seconds 1200 --generation-smoke --generation-timeout-seconds 1200 --generation-prompt "Cachepawl bounded vanilla vLLM baseline." --max-new-tokens 8
```

## Status

- status: `completed`
- reason: `bounded vanilla vLLM generation smoke completed`
- primary model: `Zyphra/Zamba2-2.7B-instruct`
- fallback model: `tiiuae/Falcon-H1-1.5B-Instruct`
- pinned vLLM: `0.21.0`
- isolated venv: `/tmp/vllm-cachepawl-venv`
- infrastructure decision: `fix-local-wsl2-gpu-nvml-first`
- model load smoke enabled: `false`
- bounded generation smoke enabled: `true`

`manifest.json` preserves separate `model_load_smoke` and
`bounded_generation_smoke` sections when both stages have been captured.

Bounded generation smoke metrics from the current artifact:

- prompt tokens: `13`
- generated tokens: `8`
- elapsed seconds: `43.399474211997585`
- tokens/sec: `0.18433403042906996`
- available GPU memory after generation: `10905399296`
- peak GPU memory: unavailable from the parent process because vLLM runs the
  engine in a child process

This step does not add vLLM to the main Cachepawl environment, run long-lived
serving, monkeypatch vLLM, replace allocators, add Triton kernels, add copy
kernels, add LSDR, or run real model quality evaluation.
