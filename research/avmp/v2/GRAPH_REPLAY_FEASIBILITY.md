# CUDA graph replay feasibility for `TritonAVMPAllocator`

## §1 Goal

PR #49 (slowdown root cause) showed `TritonAVMPAllocator` pays ~75 µs CPU per `allocate()` vs ~7 µs for Python baseline. PR #49 §5 proposed option **(A) CUDA graph replay** to amortize the per-launch Triton dispatch + `cuLaunchKernelEx` driver round-trip, with the claim that "~15× per-call reduction, closing the goodput gap from 0.24× to ~0.85×."

This study tests whether the claim holds on real hardware before committing to a Week-2 implementation. The follow-up task spec from the user includes an explicit STOP-on-failure gate:

> *Phase 5: latency benchmark shows graph replay ≥5x faster than Week 1 launch. […] If any phase reveals graph replay does NOT close the gap, STOP and write findings document; do not push broken code.*

For Week 1's 75 µs baseline, the 5× gate is **≤15 µs per call**. This document reports whether (A) hits it.

## §2 Method

Probe script: `research/avmp/v2/results/probe_graph_replay.py` (committed alongside this doc; runs in ~30 s on RTX 3060).

Hardware / software: RTX 3060 12 GiB, CUDA 13.0, PyTorch 2.12.0+cu130, Triton 3.7.0. Page size 64 KiB (matches Week 1 latency-bench page-size #2 and the Jamba-1.5-mini KV page). `BLOCK_SIZE = 1024` (matches Week 1 kernel constexpr). Each variant runs N = 5,000 iterations, reports p50 and p95 of per-call wall-clock latency (`time.perf_counter_ns()` around the call, no `torch.cuda.synchronize()` unless the variant name says otherwise — matching the simulator's no-sync regime per `runner.py:_timed_allocate`).

Variants tested:

| # | Description |
|---|---|
| V1 | `device_offset[0] = N` followed by a direct (no-graph) kernel launch. The naive non-graph baseline; same kernel signature as `zero_with_offset_ptr` defined in the probe (reads offset from a 1-element int64 device tensor via `tl.load`). |
| V2 | Pinned-host `host_offset[0] = N` followed by `device_offset.copy_(host_offset, non_blocking=True)` followed by direct kernel launch. Tests whether replacing the PyTorch scalar-write dispatcher path with an explicit 8-byte H2D `cudaMemcpyAsync` helps. |
| V3 | Capture the V2 pattern (the `copy_` AND the kernel) into a `torch.cuda.CUDAGraph`. Per call: pre-write `host_offset[0] = N`, then `g.replay()`. **This is the design implied by PR #49 (A).** |
| V3-sync | V3 with `torch.cuda.synchronize()` after each `g.replay()`. Documents the cost of the sync that a real consumer might add. |
| V4 | Static-offset graph: capture once with the offset baked in, replay without any pre-write. Establishes the irreducible `g.replay()` floor. |

The kernel itself is `zero_with_offset_ptr` (Triton 3.x, `@triton.jit`), reading the offset from a `tl.load(offset_ptr)` so the captured graph can replay against an offset tensor that's been updated between replays.

## §3 Results

From `uv run python research/avmp/v2/results/probe_graph_replay.py` on the dev RTX 3060:

| Variant | p50 (µs) | p95 (µs) | vs 15 µs gate | vs Week 1 (~75 µs) |
|---|---|---|---|---|
| V1 (`tensor[0]=N` + direct kernel, no sync) | **112.23** | 279.56 | ❌ 7.5× over | slower |
| V2 (pinned host `copy_` + direct kernel, no sync) | **93.77** | 222.18 | ❌ 6.3× over | slower |
| **V3 (graph replay, pinned pre-write, no sync — Strategy A)** | **53.80** | 414.16 | **❌ 3.6× over** | **~1.4× faster, NOT 5×** |
| V3-sync (V3 + per-call `cuda.synchronize()`) | **129.41** | 390.08 | ❌ 8.6× over | slower |
| **V4 (static-offset graph replay only, no sync — floor)** | **7.00** | 12.40 | **✓ meets gate** | **~11× faster** |

A second run from the same script produced V3 p50 of 70.11 µs (prior probe) and V3 p50 of 53.80 µs (this run). Both are in the same regime: per-call cost dominated by Python pre-write + captured `copy_` on every replay, ~7-10× the graph-replay floor.

## §4 Analysis: where the per-call cost goes

The per-call decomposition follows directly from V3 minus V4:

- **V4 = 7.00 µs**: irreducible CPU cost of `g.replay()` itself — the driver receives a "replay this captured graph" command and dispatches the captured kernel-launch node. Meets the gate.
- **V3 − V4 = 46.80 µs**: cost added by making the offset varyable. Specifically:
  - `host_offset[0] = N` on a pinned 1-element int64 tensor: PyTorch's dispatcher overhead for a scalar item-write, ~10-30 µs (slow despite the tensor being on CPU).
  - `device_offset.copy_(host_offset, non_blocking=True)` re-played as a captured graph node: the captured copy launches a small kernel under the hood, ~15-30 µs per replay.
  - Per-call Python overhead in the test loop: ~1-2 µs.

V1 vs V2 (112.23 vs 93.77 µs) shows the pinned-host `copy_` saves only ~18 µs over the naive `device_tensor[0] = N`. Both still pay the full Triton Python launcher cost (the kernel is OUTSIDE any graph in V1/V2).

V3 vs Week 1 (54 vs 75 µs) shows graph replay saves the Triton launcher cost (~21 µs net) but the gain is eaten by the per-call host write + replayed `copy_`. **Net win: ~1.4× per call, not the 5× the gate requires and far below the ~15× PR #49 §5 claimed.**

**Why V4 meets the gate but is unusable.** V4 captures with a STATIC offset baked into the kernel-launch node. To use that pattern for varying offsets, you'd need one captured graph per distinct offset value. A 4 GiB pool with 64 KiB pages has 65,536 distinct offsets; at ~10 KB of host + GPU graph metadata per capture (rough estimate from PyTorch internals), that's ~650 MiB of graph memory, growing unboundedly across the sweep. **Infeasible.**

**The bottleneck is not the kernel and not graph replay** — it's the per-call communication of a single integer from CPU to GPU. CUDA's `cudaGraphExecKernelNodeSetParams` would let us update the captured kernel's arguments at replay time without a tensor write, but PyTorch does not expose this in its Python API as of 2.12.0. The Triton ecosystem may add support for "argument-update on replay" in future releases; until then, Strategy A's per-call cost is bounded below by the offset-communication cost (~30-50 µs in practice), not by the replay machinery.

## §5 Decision

**Do not pursue Strategy A as specified.** It produces a ~1.4× per-call improvement, not the 5× the gate requires, and not the 15× PR #49 §5 hypothesized. The probe data + the `cudaGraphExecKernelNodeSetParams` API gap argue this is not a tuning issue but a fundamental ceiling of the design.

Two paths, in priority order:

### (C) Re-frame the metric in v2 paper §5 — **recommended for v2 deadline**

Accept the Week 1 / Week 2 framing as the published result. The byte-identical event-stream parity remains the headline correctness claim (`oom_count = 510 = 510 = 510` exactly, 0.0000% drift on `effective_batch_size_p50` across 36 paired seed-cells). The Week 1 latency benchmark's batched-amortized 45-116 µs/call is the per-allocate throughput evidence. The simulator's goodput ratio (0.237×) is a per-call-orchestration artifact that does not reflect inference throughput; a real engine queues many launches per decode step and pays the per-call cost once per decode step, not once per allocate. Paper §5 needs one paragraph explaining this, citing PR #49 (root cause) and this study (graph replay does not help).

Zero engineering effort. Honest. Defensible under review.

### (B-redesigned) Batched / deferred kernel-launch surface — **defer to v2.1**

The throughput-positive design is to amortize the per-call cost across N allocations: pre-fill an N-slot offset buffer in pinned memory, capture a graph that launches N kernels reading N consecutive slots, replay the graph once per N allocations. Per-allocation cost approaches V4 floor (~7 µs / N).

This requires changes the current task doesn't scope:

1. A `flush()` or `commit()` boundary on the allocator API that the consumer (inference engine, simulator) calls between decode steps.
2. A ring buffer protocol that maps `allocate()` calls to slots and re-captures the graph when the page count varies.
3. Updated correctness reasoning: a freshly allocated page must be zeroed BEFORE the consumer reads it. With a per-decode-step flush, this holds for attention/SSM reads at the next decode step boundary — but a parity test must check that no consumer reads the page between allocate() return and flush().
4. Updated `parity_smoke` test that calls `flush()` at the right moment and re-verifies 0% drift.

Plausibly 1-2 weeks of design + implementation. Open as a separate task / RFC once v2 ships with framing (C).

### What NOT to do

- **Do not** implement Strategy A with the current per-call cost — it would land code that does not deliver the performance the PR description claims, and the parity tests would pass but the goodput ratio would barely move (~0.34× instead of 0.24×; still well below 0.85×).
- **Do not** change the simulator harness to exclude Triton overhead from `wall_s` — that is moving the goalposts; the per-call cost is real Python work that any naive consumer would pay.
- **Do not** lower the 5× gate — the gate exists to prevent shipping a "graph replay enabled" feature flag that does not measurably help. Lowering it to 1.4× would let the design land but mislead readers about what graph replay buys.
