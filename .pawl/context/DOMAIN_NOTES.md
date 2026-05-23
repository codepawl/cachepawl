# Domain Notes

This file captures domain-specific vocabulary, assumptions, and concepts that coding agents need to avoid misunderstanding the project.

## Key Concepts

- AVMP means asymmetric virtual memory paging: separate native physical pools for KV pages and SSM blocks, exposed through virtual handles.
- v1 Python AVMP is the published research prototype.
- v2 Triton work validates hardware-realization correctness, but its per-allocate Python orchestration overhead keeps production deployment deferred to v2.1.
- The active vLLM plan uses Python `AsymmetricVirtualPool`, not `TritonAVMPAllocator`, for the end-to-end demo.

## Important Terms

- KV page: variable-length attention cache allocation unit.
- SSM block: fixed-size Mamba/state-space model state allocation unit.
- Path C: vLLM integration path that subclasses `KVCacheManager` and injects via scheduler configuration.
- Path A: fallback vLLM fork path if Path C cannot cleanly override internals.

## Non-Obvious Constraints

- Always capture a vanilla vLLM baseline before claiming AVMP deltas.
- Keep paper-facing claims aligned with measured artifacts under `research/avmp/v2/`.
- Do not revive per-allocate Triton production deployment before the v2.1 batched/deferred API work.

## Important Files

- `research/avmp/v2/VLLM_INTEGRATION_ROADMAP.md`
- `research/avmp/v2/VLLM_INTEGRATION_AUDIT.md`
- `research/avmp/v2/VLLM_DEV_SETUP.md`
- `research/avmp/v2/TRITON_ROADMAP.md`
