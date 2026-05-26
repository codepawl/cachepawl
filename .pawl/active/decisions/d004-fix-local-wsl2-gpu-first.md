# D004 — Use restored local WSL2 GPU visibility for vLLM baseline

Status: Accepted
Created: 2026-05-23
Updated: 2026-05-26
Completed: N/A
TTL: Keep while active
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Decision

For Sprint 1 / T001 runtime baseline capture, use the restored local WSL2
GPU/NVML path before moving the baseline to another machine.

## Reason

The current host is the intended RTX 3060 development target. The runtime gate
was previously blocked:

- `nvidia-smi` exits 255 and reports GPU access blocked by the operating system.
- PyTorch reports CUDA unavailable and zero CUDA devices.
- The active Cachepawl uv environment does not have vLLM installed.

As of 2026-05-23, local GPU/NVML visibility is restored outside the sandbox:
`nvidia-smi` reports an NVIDIA GeForce RTX 3060 with 12288 MiB VRAM, and torch
reports CUDA available with one CUDA device. The isolated pinned vLLM
environment at `/tmp/vllm-cachepawl-venv` imports `vllm==0.21.0`, sees CUDA,
and completed a bounded vanilla model-load smoke for
`Zyphra/Zamba2-2.7B-instruct`.

As of 2026-05-26, future pinned vLLM runtime work should prefer the durable
environment at `~/.cache/cachepawl/vllm-cachepawl-venv`. The old `/tmp`
environment path is no longer the primary runbook path and may leave stale
compiled-extension/source-path state behind.

## Consequences

- Future runtime baseline and observation work should use
  `~/.cache/cachepawl/vllm-cachepawl-venv` with
  the pinned vLLM environment documented in
  `research/avmp/v2/VLLM_DEV_SETUP.md`.
- Do not add vLLM to the main Cachepawl environment.
- Keep `research/avmp/v2/results/vllm-baseline/` as the structured artifact
  location for readiness, blocker, and bounded runtime-smoke evidence.
- Separate Linux GPU and rented cloud GPU paths remain deferred.

## Alternatives Considered

- Use a separate Linux GPU machine.
- Use a rented cloud GPU.
- Keep T001 runtime baseline blocked without selecting a repair path.

## Date

2026-05-23
