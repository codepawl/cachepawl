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

## Stub kept for compatibility

`benchmarks/dummy_cache_workload.py` predates the harness and will host
a longer driver example once the first concrete allocator lands. It is
not on any test path.
