# vLLM Runtime Baseline Capture

This directory records the measurement-only vanilla vLLM baseline status for
Sprint 1 / T001.

## Contents

- `baseline.jsonl`
- `manifest.json`
- `README.md`

## Reproduction

```bash
PYTHONPATH=src /home/nxank4/.cache/cachepawl/vllm-cachepawl-venv/bin/python benchmarks/scripts/capture_vllm_baseline.py --output-dir /tmp/cachepawl-proof-current-len1024-seq1 --model "Zyphra/Zamba2-2.7B-instruct" --fallback-model "tiiuae/Falcon-H1-1.5B-Instruct" --timestamp "2026-06-05T01:00:00Z" --max-model-len 1024 --gpu-memory-utilization 0.7 --max-num-seqs 1 --gpu-total-bytes 12884901888 --gpu-name "NVIDIA GeForce RTX 3060" --runtime-timeout-seconds 1200 --generation-smoke --generation-timeout-seconds 60 --generation-prompt "Short prompt." --max-new-tokens 1
```

## Status

- status: `not_runnable`
- reason: `torch reports CUDA unavailable`
- primary model: `Zyphra/Zamba2-2.7B-instruct`
- fallback model: `tiiuae/Falcon-H1-1.5B-Instruct`
- pinned vLLM: `0.21.0`
- isolated venv: `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv`
- infrastructure decision: `fix-local-wsl2-gpu-nvml-first`
- model load smoke enabled: `false`
- bounded generation smoke enabled: `true`

`manifest.json` preserves separate `model_load_smoke` and
`bounded_generation_smoke` sections when both stages have been captured.

This step does not add vLLM to the main Cachepawl environment, run long-lived
serving, monkeypatch vLLM, replace allocators, add Triton kernels, add copy
kernels, add LSDR, or run real model quality evaluation.
