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

Fresh install and smoke check:

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cpu
uv run cachepawl --help
uv run cachepawl diagnose-vllm --help
```

Basic package import:

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

## vLLM Diagnostic CLI

Use `cachepawl diagnose-vllm` to turn an existing translated vLLM runtime cache
observation into an advisory report. This command consumes artifacts; it does
not import vLLM, load a model, require CUDA/NVML, modify vLLM, monkeypatch, or
replace allocators.

```bash
uv run cachepawl diagnose-vllm \
  --translated-cache-config research/avmp/v2/results/vllm-runtime-cache-plan-observation/translated_runtime_cache_config.json \
  --raw-safe-metadata research/avmp/v2/results/vllm-runtime-cache-plan-observation/raw_safe_metadata.json \
  --output-dir research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli
```

`--translated-cache-config` is required. `--raw-safe-metadata` is optional, but
include it when available so the report can include safe runtime metadata. The
command writes `report.json`, `summary.md`, and `manifest.json`.

For CI or release checks, the command can also print the generated summary and
fail when advisory metrics cross local gates:

```bash
uv run cachepawl diagnose-vllm \
  --translated-cache-config research/avmp/v2/results/vllm-runtime-cache-plan-observation/translated_runtime_cache_config.json \
  --raw-safe-metadata research/avmp/v2/results/vllm-runtime-cache-plan-observation/raw_safe_metadata.json \
  --output-dir /tmp/cachepawl-diagnostic \
  --summary-only \
  --format markdown \
  --fail-on-waste-fraction 0.5 \
  --fail-on-overestimation-ratio 2.0
```

`--summary-only` prints the generated summary to stdout after writing the same
output files. `--format` controls that stdout payload (`markdown` for
`summary.md`, `json` for `report.json`). Threshold flags return exit code `1`
only when the reported `wasted_fraction` or `overestimation_ratio` is greater
than the configured value. Input and schema errors continue to return exit code
`2`.

This artifact-input mode requires no vLLM dependency, CUDA, GPU, or NVML. It
does not rerun vLLM, load a model, monkeypatch, replace allocators, or change
vLLM behavior. Output is advisory-only; runtime memory savings require a future
mutation hook.

Expected output for the current committed diagnostic example:

```text
classification: planner_advisory_available
advisory_only: true
observed_reserved_bytes: 2910781440
observed_useful_bytes: 1679258112
cachepawl_recommended_bytes: 1679258112
advisory_savings_bytes: 1231523328
overestimation_ratio: 1.7333734577189286
wasted_fraction: 0.4230902777777778
```

The same values appear in `report.json`:

- `observed_reserved_bytes`: 2,910,781,440
- `observed_useful_bytes`: 1,679,258,112
- `cachepawl_recommended_bytes`: 1,679,258,112
- `advisory_savings_bytes`: 1,231,523,328
- `overestimation_ratio`: 1.7333734577189286
- `wasted_fraction`: 0.4230902777777778

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
