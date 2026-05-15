# AVMP v1 preview sweep

Committed `--quick` reference data showing `avmp_static_mr05` running
alongside the three baseline variants on `uniform_short`. The data
ships with the harness-integration PR that registers `avmp_static`.

## What this preview shows

One sweep cell per variant:

- `uniform_short` x `jamba_1_5_mini` x 1 GiB x seed 1 (the `--quick`
  preset, four cells total).

Headline numbers (see `quick/report.md` for the full table):

| variant            | frag_during_load | frag_peak | OOMs | kv_free_MiB | ssm_free_MiB |
|--------------------|-----------------:|----------:|-----:|------------:|-------------:|
| padded_unified     | 0.682            | 0.995     |    0 |           - |            - |
| fixed_dual_mr05    | 0.202            | 0.961     |    5 |     512.000 |      512.000 |
| fixed_dual_mr09    | 0.219            | 0.969     |  247 |     102.375 |      921.500 |
| **avmp_static_mr05** | **0.202**      | **0.961** |  **5** | **512.000** | **512.000** |

`avmp_static_mr05` lands byte-identical to `fixed_dual_mr05` on every
deterministic metric in this sweep. That is the expected v1 outcome:
the static partition is the same physical layout, the LRU-with-cross-
pool-isolation eviction policy is the same, so the numbers match. The
contribution this PR ships is the virtual-handle abstraction over that
partition, not a new behavior. It is API surface plus a
use-after-free-safe handle id space, ready for v2 to build on.

## What this preview does NOT show

- AVMP's rigidity weakness against ratio mismatch. Only `mr05` ships
  here; `mr09` would surface the same 247-OOM datapoint `fixed_dual`
  shows, because v1's static partition is just as rigid as `fixed_dual`
  at `mr=0.9`. That cell will appear in the v2 preview, where the
  cross-pool rebalancing path can give AVMP a meaningful contrast.
- The v2 cross-pool eviction benefit. v1 reports
  `cross_pool_eviction_count = 0` in every cell.
- Triton kernel impact. AVMP v1 is pure Python plus PyTorch tensors.
- Latency comparisons. Latency is non-deterministic across reruns; the
  `aggregated_deterministic.json` subset excludes it.

## How to regenerate

```bash
uv run python -m cachepawl.benchmarks.compare --quick --device cpu \
    --output benchmarks/results/avmp-v1-preview/quick/
```

The `aggregated_deterministic.json` subset is byte-stable across reruns
at the same seed on CPU. Verified this run by diffing two consecutive
sweeps; matched exactly. The PNG figures are byte-stable at the same
matplotlib version (the existing plot-determinism tests pin this).

The `report.md` footer shows the regenerate command pointing at
`benchmarks/results/baseline/quick/`; that is the report renderer's
hardcoded default. Substitute the path above for the AVMP preview.

## Sibling data

- `benchmarks/results/baseline/quick/` is the three-variant baseline
  reference (no AVMP). Kept for historical comparison; this preview
  supersedes it for any future AVMP-bearing review.
