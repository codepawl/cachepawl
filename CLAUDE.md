# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Cachepawl is a research codebase for a hybrid KV-cache and SSM-state allocator targeting Mamba-Transformer-MoE language models. The shipping product is **pre-alpha**: most user-facing APIs (`MemoryPool`, `KVCacheManager`, `StateCacheManager`, `HybridCacheCoordinator`) raise `NotImplementedError`. The actively developed part is the **AVMP (Asymmetric Virtual Memory Paging)** allocator under `src/cachepawl/allocator/avmp/` and its benchmark harness under `src/cachepawl/benchmarks/`. An accompanying ACM paper draft lives under `research/avmp/`.

Two RFC-style design docs anchor the architecture: `docs/designs/0001-asymmetric-virtual-memory-paging.md` (AVMP core) and `docs/designs/0002-dynamic-pool-rebalancing.md` (runtime pool migration). Read these before touching anything under `src/cachepawl/allocator/avmp/`.

## Commands

Environment is managed by **uv** (not pip directly). Python 3.10 minimum; CI matrixes 3.10 and 3.12.

```bash
# Install (GPU)
uv sync
# Install (CPU-only — laptops, CI, Codespaces)
uv sync --extra-index-url https://download.pytorch.org/whl/cpu

# Lint and format (CI runs both)
uv run ruff check .
uv run ruff format --check .

# Type-check (strict mode; see pyproject.toml [tool.mypy])
uv run mypy src/cachepawl tests research/avmp/scripts

# Test
uv run pytest tests/unit -v                          # full unit suite
uv run pytest tests/unit/benchmarks -v               # one subtree
uv run pytest tests/unit/path/test_x.py::test_name   # single test
uv run pytest -m gpu                                  # CUDA-only tests
```

CI (`.github/workflows/ci.yml`) runs ruff check, ruff format check, mypy on the matrix, and `pytest tests/unit` on the matrix. No integration suite gates merge.

### Running the benchmark harness

```bash
# Single benchmark run
uv run python -m cachepawl.benchmarks.run \
    --workload {uniform_short,mixed_long,agentic_burst} \
    --allocator <registered-name> \
    [--device {cpu,cuda}] [--output benchmarks/results/] [--seed N]

# Multi-cell parameter sweep with aggregation + report + figures
uv run python -m cachepawl.benchmarks.compare \
    --variant-set {baseline,batch_size_sweep,threshold_sweep_stage2,throughput_v2} \
    --device {cpu,cuda} [--max-total-bytes N] \
    --output benchmarks/results/<dir>/

# Quick 5-cell smoke
uv run python -m cachepawl.benchmarks.compare --quick --device cpu --output /tmp/sweep
```

Post-sweep analysis CLIs:

```bash
uv run python -m cachepawl.benchmarks.analysis.lexicographic_rank \
    --input <aggregated.json>
uv run python -m cachepawl.benchmarks.analysis.throughput_analysis \
    --input <aggregated.json> --output throughput_analysis.md
```

### Building the AVMP paper

From `research/avmp/`:

```bash
make figures   # regenerate from committed sweep JSONs
make tables    # regenerate table fragments
make paper     # pdflatex + bibtex → paper.pdf
make all       # figures → tables → class extract → paper
make clean
make arxiv     # arxiv-submission.tar.gz
```

`make class` is a one-time bootstrap that extracts `acmart.cls` from the vendored `acmart.dtx`; the resulting `.cls`/`.cfg` files are gitignored.

## Architecture

### Allocator layer (`src/cachepawl/allocator/`)

`base.py` defines the narrow `Allocator` ABC (`allocate`, `free`, `stats`) and `AllocatorStats`. Every concrete allocator implements this contract. Three implementations exist:

- `baselines/padded_unified.py` — vLLM `HybridKVCacheCoordinator` mirror. Single unified pool, pads SSM state to KV page size. Surfaces `padding_waste_bytes` etc. in `allocator_specific_stats`. Used as the "rigidity" worst-case baseline.
- `baselines/fixed_dual.py` — SGLang static dual-pool mirror. Two physical pools sized at construction via `mamba_ratio`. Convention: `mamba_ratio=0.9` matches SGLang's `--mamba-full-memory-ratio` default (90% to SSM). Documented in `benchmarks/README.md` under "mamba_ratio convention".
- `avmp/` — the research contribution. `VirtualHandle` (opaque 32-bit tagged handle) → `VirtualPageTable` → `KVPagesStore` / `SSMBlocksStore`. `AsymmetricVirtualPool` is the wrapper exported via `cachepawl.allocator.avmp`. `avmp_static` keeps the partition frozen; `avmp_dynamic` adds a state-machine-driven runtime migration triggered inside `CapacityError` handling (not at a pre-emptive sampling hook — see RFC 0002 §4.2 for why).

Allocators may optionally implement two duck-typed protocols the runner detects via `isinstance(obj, _AllocatorContextProto)` and `_AllocatorStatsExporter`: `set_current_layer_kind` / `set_current_request_id` (for context propagation) and `get_allocator_stats() -> Mapping[str, float]` (for allocator-specific metrics that flow into `AllocatorMetrics.allocator_specific_stats`).

### Cache layer (`src/cachepawl/cache/`)

Interfaces only. `KVCacheManager`, `StateCacheManager`, `HybridCacheCoordinator` exist as stubs raising `NotImplementedError` with messages pointing at the unblocking design milestone. Do not "implement" these speculatively — they gate on the design discussion in `docs/architecture.md` and `docs/design-rationale.md`.

### Benchmark harness (`src/cachepawl/benchmarks/`)

Three sub-packages with distinct concerns:

- **`harness/`** — event-driven simulation engine. `runner.run_benchmark()` walks a min-heap of `(tick, kind, request_id)` events (arrival, growth, departure) and drives an `Allocator`. `MetricsCollector` (context manager) captures peak occupancy, latency samples, fragmentation samples, active-request samples, and per-request OOM flags for strict completion accounting. `BenchmarkRun` is the JSON-serialized result; **`SCHEMA_VERSION` lives in `schema.py`** and is currently `"1.2.0"`. Older artifacts (1.0.0, 1.1.0) deserialize with defaults via `_pop_*_with_default` helpers — keep this backward-compat path intact when adding fields.

  After the `with MetricsCollector` block exits, the runner MUST call `collector.finalize_throughput_metrics()` to populate the derived fields (`effective_batch_size_*`, `goodput_requests_per_second`, `completion_ratio`, `time_to_first_oom_seconds`). The strict completion semantic — "a request counts as completed iff NO OOM occurred during its lifetime AND every block freed cleanly" — is enforced by the runner threading a `request_had_oom: dict[int, bool]` through `_process_arrival`, `_timed_allocate`, `_process_growth`, and `_process_departure`.

- **`compare/`** — multi-cell sweep + aggregation + report rendering. `sweep.py` defines `AllocatorVariant` and four named variant presets (`DEFAULT_VARIANTS`, `BATCHSIZE_SWEEP_VARIANTS`, `THRESHOLD_SWEEP_STAGE2_VARIANTS`, `THROUGHPUT_V2_VARIANTS`); add new presets next to those rather than mutating existing ones (reproducibility of committed snapshots). `aggregate.py` produces `AggregatedRow` (median across replicates) with a `deterministic_subset()` projection used by reproducibility byte-diff tests — **only event-driven metrics belong in the subset**; wall-clock fields (goodput, time_to_first_oom, allocator-specific `*_ns` counters) stay out. `report.py` and `plots.py` consume `AggregatedMetrics`; the new throughput plots emit both PDF (paper) and PNG (review) via the `_save_dual_format` helper.

- **`analysis/`** — post-sweep analyzers. `lexicographic_rank.py` ranks variants on the 3-level key `(total_oom asc, effective_batch_size_p50 desc, fragmentation asc)` with OOM tie-tolerance. `throughput_analysis.py` evaluates the pre-registered Tier 1 PR B stop rule (`effective_batch_size_p50` ≥ 1.05× baseline on ≥2 workloads OR `goodput` ≥ 1.10× on ≥1 workload) and emits a markdown verdict regardless of outcome.

The harness registers allocators at import via the `REGISTRY` dict in `cachepawl.benchmarks.__init__`. Two baselines (`padded_unified`, `fixed_dual`) plus the AVMP variants are pre-registered. Use `register_allocator(name, factory)` for custom allocators.

### Committed sweep artifacts (`benchmarks/results/`)

Reference snapshots are checked in — `avmp-v1-preview/`, `avmp-v2/`, `avmp-v2-r2/`, `avmp-v2-gpu/`, `avmp-v2-batchsize-sweep/`, `avmp-v2-threshold-sweep/`, `avmp-v2-throughput/`. Each ships `aggregated.json` + `aggregated_deterministic.json` + `report.md` + `figures/` + `SWEEP_METADATA.json`. **Per-cell `runs/` directories are gitignored** for newer sweeps (see `.gitignore` lines around `avmp-v2-throughput/full/runs/`). The `aggregated_deterministic.json` files are byte-stable across CPU reruns at the same seed and matplotlib version; the `paper.tex` figures and tables generation scripts under `research/avmp/scripts/` consume them.

When the sweep grid grows (or `deterministic_subset()` changes), committed `aggregated_deterministic.json` files become stale and need regeneration. No test reads them directly, so CI does not break — they are reproducibility checkpoints, not pinned fixtures.

### Model spec (`src/cachepawl/models/spec.py`)

`HybridModelSpec`, `LayerSpec`, `LayerKind` (ATTENTION vs MAMBA2), `AttentionLayerProfile`, `SSMLayerProfile`. Most reference configs (`MAMBA2_1B3_REF`, etc.) are `None` placeholders awaiting upstream config mapping.

## Conventions worth honoring

- **No em dashes anywhere** (commit messages, paper LaTeX, code comments). The project standard uses hyphens or commas instead.
- **No AI attribution in git** — no `Co-Authored-By` for AI tools, no `🤖 Generated with ...` footers, no `claude/` or `ai/` branch prefixes. Authors are humans.
- **Strict mypy is enabled project-wide** (`disallow_any_explicit`, `disallow_any_generics`, `strict = true`). One narrow exception: `tests/unit/allocator/stateful/*` relaxes `disallow_any_explicit` because Hypothesis's `@rule` / `@invariant` decorators carry `Any`.
- **Schema bumps**: when changing `AllocatorMetrics`, bump `SCHEMA_VERSION` (minor for additive, major for breaking), add a `_pop_*_with_default` helper for the new field, and add a backward-compat test in `tests/unit/benchmarks/test_schema_migration.py`. Document the migration in the schema-version table of `benchmarks/README.md`.
- **Variant set immutability**: do not mutate the named variant presets in `compare/sweep.py` — committed reports reference them by name. Add a new preset and CLI choice instead.
- **`mamba_ratio` semantics**: 0.9 means **90% to SSM, 10% to KV** (matches SGLang's `--mamba-full-memory-ratio`, not the inverse). Pinned by `test_mamba_ratio_09_assigns_90_percent_to_ssm_pool`.
- **Paper layout gotcha (`\flushbottom`)**: `acmart` sigconf defaults to `\flushbottom` (acmart.cls:823-836), which stretches vertical glue across paragraph and section breaks so both columns end at the same y. When a page does not naturally fill both columns, that glue accumulates as a large blank gap at a section transition (we hit this between §1.1 and §1.2: ~5 lines of empty space). Symptom: `\vspace{-Xex}`, `\renewcommand\subsection` with smaller before-skip, and raw `\vskip-30pt` ALL appear to do nothing (pixel-identical render). The gap is not `\subsection`'s before-skip; it is column-balancing glue. Fix: `\raggedbottom` in the preamble. Diagnose by rendering page 1 to PNG at scale=3 and `ImageChops.difference` between two variants; if the bbox is `None`, you are touching the wrong knob.
- **`make paper` only works from `research/avmp/`**: there is no top-level `paper` target. If you `cd` away and re-run, `make` says `No rule to make target 'paper'` and you might miss it while skimming output. Always confirm `pwd` before reading "Output written" lines as authoritative; old log files lying around can look like fresh build output.

## Hardware

Reference machine for GPU sweeps: single RTX 3060 12 GiB on Linux (WSL2 supported). CI runs CPU-only and the harness falls back to `Allocator.stats()` for fragmentation when CUDA is unavailable. AVMP's design carries a 2× physical-footprint cost (RFC 0002 §4.3): on a 4 GiB pool it peaks at ~9 GiB reserved. `--max-total-bytes 4294967296` caps cells when needed.
