# Cachepawl

[![arXiv](https://img.shields.io/badge/arXiv-2605.22416-b31b1b.svg)](https://arxiv.org/abs/2605.22416)

> Pre-alpha. Core allocator prototypes, benchmark harnesses, and research artifacts exist,
> but runtime integrations and public APIs remain provisional.

Cachepawl is a hybrid cache allocator for next-generation language models that
mix attention, state-space (SSM/Mamba), and mixture-of-experts layers. Existing
KV cache managers (vLLM, SGLang, TensorRT-LLM) were built for pure transformer
stacks; they handle a uniform per-layer cache shape and an append-only access
pattern. Hybrid Mamba-Transformer-MoE models break both assumptions at once:
attention layers want variable-length KV blocks, SSM layers want fixed-size
state blocks, and MoE routing turns request shape into a runtime decision.

Cachepawl is the experiment to fix that gap: one allocator that owns a single
VRAM budget and serves both cache kinds without leaving most of the device idle.

## Target architectures

The initial design targets these published hybrid models:

- Mamba-2
- Jamba
- Zamba2
- Samba
- Hymba
- RecurrentGemma

Reference configs live in `src/cachepawl/models/spec.py` and are intentionally
left as `None` placeholders until each upstream config is mapped in.

## Install

Requires Python 3.10 or newer. Triton is gated to Linux because there are no
upstream Triton wheels for macOS or Windows.

```bash
uv sync
```

For environments without a CUDA GPU (CI, laptops, Codespaces) use the CPU-only
torch index to keep wheel size down:

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cpu
```

## Quickstart

```python
import cachepawl

print(cachepawl.__version__)
```

The research allocator paths are implemented under `allocator/avmp/` and the
benchmark harness is available under `benchmarks/`. Some public base interfaces
are still placeholders: calls into `MemoryPool`, `KVCacheManager`,
`StateCacheManager`, `HybridCacheCoordinator`, `vram_info`, and
`cuda_capability` will raise `NotImplementedError` with a message pointing at
the design milestone that unblocks them.

## Layout

```
src/cachepawl/
  allocator/   block pool and eviction policy interfaces
  cache/       KV, SSM state, and hybrid coordinator managers
  kernels/     reserved for Triton kernels
  integrations/ optional runtime integration scaffolds
  models/      hybrid model layout descriptors
  quant/       cache element dtypes (FP16, BF16, INT8, FP8, FP4)
  utils/       device and VRAM helpers
```

## Paper

The AVMP allocator is described in [arXiv:2605.22416](https://arxiv.org/abs/2605.22416). Source under `research/avmp/`.

## Documentation

- [docs/architecture.md](docs/architecture.md): two-pool vs unified-pool tradeoff and prior art in vLLM and SGLang.
- [docs/design-rationale.md](docs/design-rationale.md): why hybrid Mamba-Transformer-MoE workloads break existing cache solutions.
- [benchmarks/README.md](benchmarks/README.md): benchmarking strategy and target hardware.

## Status

- **v1** (Python prototype): published as [arXiv:2605.22416](https://arxiv.org/abs/2605.22416). Source under `research/avmp/`.
- **v2** (Triton hardware realization): correctness oracle complete via PRs [#47](https://github.com/codepawl/cachepawl/pull/47) (kernel + smoke), [#48](https://github.com/codepawl/cachepawl/pull/48) (full 216-cell sweep at 0% parity drift). The performance investigation is closed: per-allocate Python orchestration overhead is documented in [`research/avmp/v2/SLOWDOWN_ROOT_CAUSE.md`](research/avmp/v2/SLOWDOWN_ROOT_CAUSE.md) (PR [#49](https://github.com/codepawl/cachepawl/pull/49)), and the CUDA graph replay mitigation is shown infeasible under PyTorch 2.12.0 in [`research/avmp/v2/GRAPH_REPLAY_FEASIBILITY.md`](research/avmp/v2/GRAPH_REPLAY_FEASIBILITY.md) (PR [#50](https://github.com/codepawl/cachepawl/pull/50)). The hardware-realization narrative for the v2 paper lives at [`research/avmp/v2/PAPER_SECTION_5_DRAFT.md`](research/avmp/v2/PAPER_SECTION_5_DRAFT.md). Production deployment of the per-allocate kernel via a batched/deferred allocator API is deferred to v2.1.
- **vLLM integration**: in progress for the ML for Systems @ NeurIPS 2026 workshop. The v2 paper's core performance contribution.

APIs may change without notice until the integration lands. Track progress in the issues queue.

## License

MIT. See [LICENSE](LICENSE).
