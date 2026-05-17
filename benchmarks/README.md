# Benchmarks

Two-phase microbenchmark plan for hybrid KV plus SSM cache allocators.

1. **Synthetic micro-benchmarks.** No model dependency, no tokenizer, no
   logits. The harness in `cachepawl.benchmarks` drives a registered
   `Allocator` through a deterministic, seeded workload and emits one
   JSON result per run.
2. **Toy hybrid models** (later, not in this PR). 130M to 300M parameter
   hybrid models that fit on a single RTX 3060 12GB and complete a full
   eval in minutes.

## Target hardware

Reference machine: single RTX 3060 with 12 GiB of VRAM. Larger cards
(24 GiB, 40 GiB) are welcome but not required. CI runs CPU only and the
harness's fragmentation sampling falls back to `Allocator.stats()` so the
unit tests stay green without a GPU.

## RTX 3060 12 GiB budget

The three preset workloads target staying under 10 GiB peak working set
on a non-pathological allocator using `BF16` and the Jamba-1.5-Mini
layer profile:

- Per-token KV bytes per attention layer: `2 * 8 heads * 128 head_dim * 2 = 4096 B`.
- Per-sequence SSM bytes per Mamba layer: `8192 d_inner * 16 d_state * 2 = 262144 B`.
- 4 attention layers plus 28 SSM layers per request.

Worst-case single-request footprint at 64K tokens:
`64*1024 * 4 layers * 4096 + 1 * 28 layers * 262144 = ~1.08 GiB`.
At 16 concurrent requests with a typical mixed_long average of about
12K tokens the harness peaks around 2 to 3 GiB on a clean allocator,
well under the 10 GiB target.

Operators that bump `num_requests`, swap layer profiles, or pick a
larger `dtype` than `BF16` are responsible for re-checking headroom.

## CLI

```bash
uv run python -m cachepawl.benchmarks.run \
    --workload {uniform_short,mixed_long,agentic_burst} \
    --allocator <registered name> \
    [--device {cpu,cuda}] \
    [--output benchmarks/results/] \
    [--seed N] \
    [--record-memory-snapshot] \
    [--notes "free-form string"]
```

The CLI exits 2 with `Unknown workload` or `Unknown allocator` in
`stderr` when a name does not resolve. `--device` defaults to
`cachepawl.utils.device.get_device()`. `--seed` overrides the preset's
default seed for one-off reruns. `--record-memory-snapshot` only does
anything on CUDA; it dumps a `_dump_snapshot` pickle next to the JSON.

## Preset workloads

All presets use the Jamba-1.5-Mini layer profile (4 attention layers,
28 SSM layers, GQA with 8 KV heads, 128 head dim, SSM `d_inner=8192`,
`d_state=16`) and `BF16`.

| Name           | `num_requests` | Prompt length distribution | Generation length | Arrivals |
|----------------|---------------:|----------------------------|-------------------|----------|
| uniform_short  | 512            | `U(128, 1024)`             | `U(64, 256)`      | one per virtual tick |
| mixed_long     | 256            | 80% `U(512, 4096)`, 20% `U(16384, 65536)` | `U(128, 512)` | one per virtual tick |
| agentic_burst  | 256            | `LogNormal(8.5, 1.0)` clipped to `[64, 65536]` | `U(64, 256)` | Poisson rate 10 per tick |

One virtual tick equals one decode step. Same seed produces the same
request stream byte for byte on the same numpy version.

## Result schema

Each run emits one `BenchmarkRun` JSON file at:

```
<output_dir>/<allocator_name>/<workload_name>/<UTC-timestamp>.json
```

Top-level shape (schema version `1.0.0`):

```jsonc
{
  "schema_version": "1.1.0",
  "spec": { "name": "uniform_short", "num_requests": 50, "dtype": "bf16", ... },
  "allocator_name": "mock",
  "hardware": { "device": "cpu", "gpu_name": null, ... },
  "environment": { "torch_version": "2.12.0+cpu", "numpy_version": "2.2.6", ... },
  "started_at": "2026-05-13T12:00:00Z",
  "finished_at": "2026-05-13T12:00:01Z",
  "metrics": {
    "peak_reserved_bytes": 1000,
    "peak_allocated_bytes": 184,
    "fragmentation_samples": [0.92, 0.87, ...],
    "allocate_latency_ns": [...],
    "free_latency_ns": [...],
    "allocate_latency_percentiles": {"p50_ns": 110, "p95_ns": 250, "p99_ns": 410, "max_ns": 980},
    "free_latency_percentiles":     {"p50_ns": 60,  "p95_ns": 140, "p99_ns": 220, "max_ns": 480},
    "oom_count": 0,
    "preemption_count": 0,
    "active_requests_samples": [0, 1, 2, ...],
    "allocator_specific_stats": {"padding_waste_bytes": 12345.0, "num_pages_total": 64.0, "num_pages_used": 16.0}
  },
  "notes": "growth_events=120"
}
```

Round-trip via `BenchmarkRun.from_json(path.read_text())`.

## Schema version

The current schema is `1.1.0`. Build behavior across versions:

| Reading | 1.0.0 artifact | 1.1.0 artifact | 2.x artifact |
|---|---|---|---|
| `BenchmarkRun.from_json` | accepted; `allocator_specific_stats` defaults to `{}` | accepted | rejected with `ValueError` naming the version |

`1.0.0 -> 1.1.0` change: `AllocatorMetrics.allocator_specific_stats: dict[str, float]` is the only new field. Convention is documented in the dataclass docstring: keys are strings, values are float. Counts and bytes convert to float at record time. Non-numeric tags go in `BenchmarkRun.notes`.

## Registered allocators

The default registry ships with two baselines:

| Name | Source | Pathology surfaced via `allocator_specific_stats` |
|---|---|---|
| `padded_unified` | vLLM-style `HybridKVCacheCoordinator` mirror | `padding_waste_bytes`, `num_pages_total`, `num_pages_used`, `page_size_bytes` |
| `fixed_dual` | SGLang-style static dual-pool mirror | `pool_underused_bytes_kv`, `pool_underused_bytes_ssm`, `mamba_ratio`, plus per-pool occupancy |

Each baseline retains the documented upstream weakness on purpose so that comparison data against future allocators stays honest. See the module docstrings for pinned upstream commit citations.

## REPL: register a custom allocator

The two baselines run end to end without any setup. To plug in your own allocator instead:

```python
from collections.abc import Sequence
from pathlib import Path

import torch

from cachepawl.allocator.base import Allocator, AllocatorStats
from cachepawl.benchmarks import (
    PRESETS,
    WorkloadSpec,
    register_allocator,
    run_benchmark,
)


class MockAllocator(Allocator):
    def __init__(self, spec: WorkloadSpec, device: torch.device) -> None:
        del spec
        del device
        self._next = 0
        self._free: list[int] = []
        self._live: set[int] = set()

    def allocate(self, num_blocks: int, *, dtype_bytes: int) -> list[int]:
        del dtype_bytes
        ids: list[int] = []
        for _ in range(num_blocks):
            bid = self._free.pop() if self._free else self._next
            if bid == self._next:
                self._next += 1
            self._live.add(bid)
            ids.append(bid)
        return ids

    def free(self, block_ids: Sequence[int]) -> None:
        for bid in block_ids:
            if bid in self._live:
                self._live.remove(bid)
                self._free.append(bid)

    def stats(self) -> AllocatorStats:
        return AllocatorStats(
            total_blocks=1_000_000,
            free_blocks=1_000_000 - len(self._live),
            allocated_blocks=len(self._live),
            fragmentation_ratio=0.0,
        )


register_allocator("mock", MockAllocator)
spec = PRESETS["uniform_short"]
device = torch.device("cpu")
run = run_benchmark(
    allocator=MockAllocator(spec, device),
    spec=spec,
    allocator_name="mock",
    output_dir=Path("benchmarks/results"),
    device="cpu",
)
print(run.metrics.allocate_latency_percentiles())
```

The JSON lands at `benchmarks/results/mock/uniform_short/*.json`.
The `benchmarks/results/` directory is gitignored.

## Comparison sweep

For multi-cell sweeps with replicate aggregation and report rendering:

```bash
uv run python -m cachepawl.benchmarks.compare --quick --device cpu \
    --output benchmarks/results/baseline/quick/
```

`--quick` runs three cells (one per registered baseline variant on
`uniform_short` x `jamba_1_5_mini` x 1 GiB x one seed) and writes a
markdown report, two PNG figures, raw per-cell JSON, and a
`SWEEP_METADATA.json` provenance file. Drop `--quick` for the full
3 x 3 x 2 x 3 x 3 = 162-cell sweep across all workloads, both model
specs, three pool sizes, and three seed replicates.

## Committed reference artifacts

Two snapshots are committed, both produced from the same compare CLI:

- `benchmarks/results/avmp-v1-preview/quick/` is the 4-cell `--quick`
  output (`uniform_short` x `jamba_1_5_mini` x 1 GiB x 1 seed,
  4 variants). Used for CI smoke and PR previews. Regenerate:
  `python -m cachepawl.benchmarks.compare --quick --device cpu
  --output benchmarks/results/avmp-v1-preview/quick/`.
- `benchmarks/results/avmp-v1-preview/full/` is the 216-cell full
  sweep (3 workloads x 2 model specs x 3 total_bytes x 3 seeds x 4
  variants). Used as the v1 production reference. The committed
  artifact ships `aggregated.json`, `aggregated_deterministic.json`,
  `report.md` (including the cross-workload summary section),
  `figures/`, and `SWEEP_METADATA.json`. The `runs/` subdirectory is
  NOT committed; regenerate to inspect per-cell JSONs. Regenerate
  command: `python -m cachepawl.benchmarks.compare --device cpu
  --output benchmarks/results/avmp-v1-preview/full/`.

`benchmarks/results/baseline/quick/` is the legacy 3-variant baseline
snapshot from the pre-AVMP phase. Kept as historical reference; the
AVMP preview directory supersedes it for any current review.

The `aggregated_deterministic.json` sibling is bit-stable across
reruns at the same seed on CPU; the rest is provenance and
visualization. Latency stats in `aggregated.json` are
machine-dependent; the hardware that produced the committed data is
named in `SWEEP_METADATA.json`.

## Known limitations of this baseline data

The committed `--quick` snapshot is a sanity-check artifact, not a
production performance characterization. Three caveats matter for anyone
reading the numbers or planning follow-up work:

- `fragmentation_peak` samples at end-of-run when `allocated_blocks`
  approaches zero, so its value reflects the late-run drain phase as
  much as the worst in-flight moment. Keep it as a coarse signal until
  the time-weighted underuse metric (see below) lands.
- `pool_free_bytes_kv` and `pool_free_bytes_ssm` are point-in-time
  snapshots at end-of-run, not a rigidity measure. On a cleanly-departing
  workload they trivially equal the pool totals. The correct metric is a
  time-weighted average of per-pool free bytes during load, sampled via
  `MetricsCollector` at every tick. That is a deliberate follow-up.
- The 247 OOMs on `fixed_dual_mr09 + uniform_short` are not a bug. They
  surface SGLang's rigidity weakness when the workload does not match
  the configured `mamba_ratio`: an SSM-heavy split starves the KV pool
  on short-prompt traffic. This is the empirical baseline AVMP must beat
  by rebalancing dynamically.

## Data sanity invariants

Every emitted metric below must satisfy its valid range. Run the
checklist against any new sweep output before trusting the numbers.
This section exists because three of the metrics in the first
committed snapshot did NOT satisfy their ranges and the bug took a
human eye to catch; the table now serves as the reviewer's checklist.

| Metric | Source | Unit | Valid range | Comment |
|---|---|---|---|---|
| `fragmentation_during_load_mean` | mean of `1 - allocated/reserved` over ticks with `active_requests_samples[i] > 0` | dimensionless | `[0, 1]` | filtered to ignore the post-teardown sample (the runner emits one final tick with `active == 0` where ratio is forced to 1.0) |
| `fragmentation_peak` | max of the same filtered series | dimensionless | `[0, 1]` | worst-case during load |
| `peak_reserved_bytes` | `torch.cuda.max_memory_reserved` on CUDA; pool's `total_blocks` on CPU | bytes (CUDA) or blocks (CPU) | `>= 0` | on CPU this is a block count, not a byte count (existing harness quirk; CUDA gives bytes) |
| `oom_count` | runner-caught `OutOfMemoryError` | count | `>= 0` | non-zero indicates pool starvation |
| `padding_waste_bytes` (padded_unified) | sum of `(page_size - logical_bytes)` per live page | bytes | `0 <= x <= peak_allocated_bytes` | snapshot at run end. Drops to 0 on a workload where every request departs cleanly; the metric is meaningful mid-run, not at teardown |
| `pool_free_bytes_kv` (fixed_dual) | `num_pages_free * page_size_bytes` for the KV table | bytes | `0 <= x <= kv_pool_total_bytes` | snapshot at run end. On a cleanly-departing workload this equals `kv_pool_total_bytes`; for rigidity comparisons, prefer `oom_count` and `fragmentation_peak` |
| `pool_free_bytes_ssm` (fixed_dual) | `num_pages_free * page_size_bytes` for the SSM table | bytes | `0 <= x <= ssm_pool_total_bytes` | same caveat as `pool_free_bytes_kv` |
| `kv_pool_total_bytes`, `ssm_pool_total_bytes` (fixed_dual) | constructor-time pool sizes | bytes | `> 0`, sum near `total_bytes` minus alignment slack | constants for the life of the allocator |
| `mamba_ratio` | constructor arg | dimensionless | `(0, 1)` exclusive | fraction of `total_bytes` assigned to the SSM pool (see "mamba_ratio convention" below) |
| `allocate_p50_ns`, `p95_ns`, `p99_ns` | `time.perf_counter_ns` | nanoseconds | `>= 0` | NOT deterministic across reruns |

Smell tests, in priority order:

- A pool's `free_bytes` must never exceed that pool's `total_bytes`. If it does, the field is a cumulator masquerading as a snapshot.
- `fragmentation > 1` or `fragmentation < 0` means the metric source is misconfigured (look for divide-by-zero shortcuts in `MetricsCollector.sample`).
- `mamba_ratio` outside `(0, 1)` is rejected at construction.
- If `oom_count > 0` while `pool_free_bytes_<other>` is non-zero, that quantifies the rigidity cost of the static partition (stranded bytes that could have served evicted requests).
- A fragmentation_during_load value of exactly 1.000 across every variant is the canonical signature of "aggregator picked the wrong tick" (the teardown sample). Sanity-check the active count alignment.

### mamba_ratio convention

`mamba_ratio` in `FixedDualPool` mirrors SGLang's `mamba_full_memory_ratio` at commit [`22012ba1`](https://github.com/sgl-project/sglang/blob/22012ba1bc2166f2280be2ad648ba732a0ff382b/python/sglang/srt/server_args.py): the argparse help string is `"The ratio of mamba state memory to full kv cache memory"` and the default is `0.9`. Higher values give MORE bytes to mamba/SSM, less to KV.

In practice:

- `mamba_ratio = 0.5`: neutral 50/50 split; the synthetic-comparison baseline this repo ships with as the constructor default.
- `mamba_ratio = 0.9`: matches SGLang's production default. **SSM-heavy, NOT KV-heavy.** The SSM pool gets 90% of `total_bytes`. On short-context, KV-dominated workloads (for example `uniform_short` with 128-1024 token prompts) this strands the KV pool and produces many OOMs - that's the legitimate SGLang-default datapoint, not a bug.
- If you want the KV-heavy endpoint (10% to SSM, 90% to KV), pass `mamba_ratio = 0.1`. The default sweep does not currently include this; add it as a follow-up if relevant.

The convention is locked by `test_mamba_ratio_09_assigns_90_percent_to_ssm_pool` in `tests/unit/allocator/baselines/test_fixed_dual.py`.

### Sweep variant sets and resume

The compare CLI supports two variant sets via `--variant-set`:

- `baseline` (default): the 5-variant `DEFAULT_VARIANTS` (padded_unified, fixed_dual_mr05, fixed_dual_mr09, avmp_static_mr05, avmp_dynamic_mr05). Used for every sub-PR baseline and the `--quick` / `--smoke` paths.
- `batch_size_sweep`: the 12-variant stage 1 set (3 baselines + 9 `avmp_dynamic_b{N}` variants over `migration_batch_size` in `{1, 2, 4, 8, 16, 32, 64, 128, 256}`). Used by `benchmarks/results/avmp-v2-batchsize-sweep/`.

Long-running parameter sweeps (multi-hour on GPU) support resume from disk: per-cell JSONs land at `{output_dir}/runs/{variant_label}/{workload_name}/{stem}.json` immediately after each cell completes. Re-invoking the sweep against the same `--output` directory reads any existing JSONs back and skips those cells. Corrupt or schema-mismatched files are treated as missing and re-run. The "RESUMED" label in the progress output identifies skipped cells.

Resume is keyed by cell stem under the variant directory; switching `--variant-set` in the same output directory is undefined behavior (use a fresh `--output` for each variant set).

### AVMP-specific invariants (avmp_static and avmp_dynamic)

For `avmp_static` and `avmp_dynamic` runs the twenty-five keys in `allocator_specific_stats` satisfy these invariants on top of the table above. The first four were the v1 contract; the rest land across the three v2 sub-PRs (RFC 0002 sections 4.2, 4.7).

v1 invariants:

- `kv_pages_used + kv_pages_free == kv_pages_total` at every recorded sample.
- `ssm_blocks_used + ssm_blocks_free == ssm_blocks_total` at every recorded sample.
- `virtual_handles_live == kv_pages_used + ssm_blocks_used`.
- `cross_pool_eviction_count == 0.0` in v1. The field exists in the stats dict so the v2 cross-pool rebalancing path can light it up without a schema bump.

v2 sub-PR 1 invariants (observability):

- `0.0 <= kv_free_ratio <= 1.0`; `0.0 <= ssm_free_ratio <= 1.0`.
- `rebalance_enabled` in `{0.0, 1.0}`.
- `threshold_low < threshold_high`, both in `(0.0, 1.0)`.
- `migration_batch_size >= 1.0`.
- `current_pressure_state_code` in `{0.0, 1.0, 2.0, 3.0}` encoding `PoolPressureState`: `BALANCED`, `KV_PRESSURED`, `SSM_PRESSURED`, `REBALANCING`. `compute_state` never returns `REBALANCING`; the pool itself transitions through that state during a migration leg.
- `rebalance_count >= 0.0`.
- `bytes_migrated_total >= 0.0`.
- `time_spent_rebalancing_ns >= 0.0`.

v2 sub-PR 2 invariants (migration mechanics):

- `bytes_wasted_to_alignment_total >= 0.0`; accumulates `(donor_bytes mod recipient_native_unit_bytes)` on each successful migration.
- `current_kv_pool_bytes + current_ssm_pool_bytes + bytes_wasted_to_alignment_total` is invariant across every successful migration (anchored at the post-construction sum, not at `total_bytes`, because initial page/block alignment shaves bytes off at construction time and that loss is NOT tracked in `bytes_wasted_to_alignment_total`).
- After a failed migration (either donor shrink rejected or recipient grow rejected and rolled back), pool sizes equal their pre-migration values and the four migration counters are unchanged.

Worked example: `block_size_bytes = 64 KiB`, `page_size_bytes = 24 KiB`, `migration_batch_size = 2`, direction `SSM_TO_KV`. Donor releases `2 * 64 = 128 KiB`. Recipient grows by `floor(128 / 24) * 24 = 120 KiB` (5 pages). Per-event waste `= 128 mod 24 = 8 KiB`. After this event, `bytes_wasted_to_alignment_total += 8 KiB`.

v2 sub-PR 3 invariants (auto-trigger):

- `auto_rebalance_skipped_throttle >= 0.0`; monotonically non-decreasing. Counts pressure observations that the pool detected but skipped because the throttle window (`min_rebalance_interval_ns`, default 1 ms) had not elapsed since the previous auto-trigger.
- Manual triggers via `trigger_manual_rebalance` bypass the throttle; only the observation-hook auto-trigger consults it. This keeps the diagnostic-use case from sub-PR 2 unchanged.
- The auto-trigger is non-deterministic under repeated runs because the throttle reads `time.monotonic_ns`. Byte-level determinism on the deterministic JSON subset is guaranteed only when `rebalance_enabled=False`; with `rebalance_enabled=True`, treat aggregated counters as approximately reproducible but not byte-stable (RFC 0002 section 8 question 5).

These invariants are pinned by `test_avmp_run_benchmark_uniform_short_cpu` in `tests/unit/benchmarks/test_avmp_harness_integration.py`, the v2 tests under `tests/unit/allocator/avmp/test_pool_v2_*.py` and `tests/unit/allocator/avmp/test_avmp_v2_*.py`, the migration tests in `tests/unit/allocator/avmp/test_pool_migration.py`, the dynamic harness integration tests in `tests/unit/benchmarks/test_avmp_dynamic_harness_integration.py`, and the stateful invariants in `tests/unit/allocator/stateful/test_avmp_stateful.py`. A failure on the committed `--quick` preview means the sweep regenerated against a buggy build; treat the table as the reviewer's checklist for future allocator PRs.

The AVMP report rows reuse the same `kv_free_MiB` / `ssm_free_MiB` columns as `fixed_dual`; per-pool free bytes are derived from `pages_free * (pool_bytes / pages_total)`. v2 adds four extra columns (`rebalance_count`, `bytes_migrated_MiB`, `throttle_skips`, `waste_KiB`) shown for AVMP rows only; baselines render dashes there.

## Stub kept for compatibility

`benchmarks/dummy_cache_workload.py` predates the harness and will host
a longer driver example once the first concrete allocator lands. It is
not on any test path.
