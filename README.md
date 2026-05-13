# Cachepawl

> Pre-alpha. Interfaces are stable enough to plan against; nothing inside is implemented yet.

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

The full allocator API is wired up but not implemented. Calls into
`MemoryPool`, `KVCacheManager`, `StateCacheManager`, `HybridCacheCoordinator`,
`vram_info`, and `cuda_capability` will raise `NotImplementedError` with a
message pointing at the design milestone that unblocks them.

## Layout

```
src/cachepawl/
  allocator/   block pool and eviction policy interfaces
  cache/       KV, SSM state, and hybrid coordinator managers
  kernels/     reserved for Triton kernels
  models/      hybrid model layout descriptors
  quant/       cache element dtypes (FP16, BF16, INT8, FP8, FP4)
  utils/       device and VRAM helpers
```

## Documentation

- [docs/architecture.md](docs/architecture.md): two-pool vs unified-pool tradeoff and prior art in vLLM and SGLang.
- [docs/design-rationale.md](docs/design-rationale.md): why hybrid Mamba-Transformer-MoE workloads break existing cache solutions.
- [benchmarks/README.md](benchmarks/README.md): benchmarking strategy and target hardware.

## Status

Pre-alpha. APIs may change without notice until the first concrete allocator
lands. Track progress in the issues queue.

## License

MIT. See [LICENSE](LICENSE).
