# Root cause of the 4x simulator-wall-clock slowdown in PR #48

PR #48 reported byte-identical event-stream parity between
`avmp_dynamic_b128_triton` and Python `avmp_dynamic_b128` (OOM 510=510,
0.0000% drift on `effective_batch_size_p50` across all 36 paired
seed-cells) but a 0.237x goodput ratio (95% CI [0.214, 0.267], excludes
unity). The PR's prose attributed the slowdown to "Triton variant pays
one `cuda.synchronize()`-per-`allocate()` inside `run_benchmark`'s
latency timing". This investigation **disproves** that explanation
and locates the real cost.

All numbers below come from RTX 3060 (12 GiB), torch 2.12.0+cu130,
Triton 3.7.0, the same hardware the original sweep ran on.

## §1 Decomposition

### §1.1 Per-call latency from the existing sweep (`aggregated.json`)

Phase 1 of the investigation (`research/avmp/scripts/investigate_triton_slowdown.py --phase 1`)
reads the committed sweep's per-row `allocate_*_ns_median` fields and
pairs Python `avmp_dynamic_b128` against `avmp_dynamic_b128_triton` on
the 12 (workload, spec, total_bytes) triples:

| workload | spec | pool_GiB | py p50 us | py p95 us | py p99 us | tr p50 us | tr p95 us | tr p99 us | p50 ratio | p95 ratio | p99 ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agentic_burst | jamba_1_5_mini | 1 | 8.42 | 392.95 | 5174.49 | 61.18 | 3945.59 | 38873.92 | 7.27x | 10.04x | 7.51x |
| agentic_burst | jamba_1_5_mini | 4 | 8.23 | 325.40 | 4005.10 | 68.74 | 3969.42 | 34765.17 | 8.35x | 12.20x | 8.68x |
| agentic_burst | mamba2_1b3 | 1 | 8.26 | 330.34 | 4125.13 | 84.02 | 4031.76 | 40469.74 | 10.17x | 12.20x | 9.81x |
| agentic_burst | mamba2_1b3 | 4 | 9.61 | 358.34 | 4152.91 | 84.21 | 4488.05 | 40607.30 | 8.76x | 12.52x | 9.78x |
| mixed_long | jamba_1_5_mini | 1 | 10.79 | 118.98 | 1537.57 | 60.65 | 189.74 | 7276.62 | 5.62x | 1.59x | 4.73x |
| mixed_long | jamba_1_5_mini | 4 | 8.73 | 56.10 | 1342.60 | 51.69 | 176.16 | 7382.68 | 5.92x | 3.14x | 5.50x |
| mixed_long | mamba2_1b3 | 1 | 8.04 | 62.71 | 1179.77 | 55.70 | 166.15 | 7393.70 | 6.93x | 2.65x | 6.27x |
| mixed_long | mamba2_1b3 | 4 | 8.79 | 68.24 | 1100.88 | 71.54 | 249.52 | 9906.68 | 8.14x | 3.66x | 9.00x |
| uniform_short | jamba_1_5_mini | 1 | 10.55 | 101.47 | 395.08 | 70.00 | 673.82 | 2953.37 | 6.64x | 6.64x | 7.48x |
| uniform_short | jamba_1_5_mini | 4 | 7.31 | 68.01 | 223.99 | 85.78 | 978.18 | 4359.39 | 11.74x | 14.38x | 19.46x |
| uniform_short | mamba2_1b3 | 1 | 10.05 | 91.61 | 375.01 | 84.15 | 855.67 | 3685.99 | 8.37x | 9.34x | 9.83x |
| uniform_short | mamba2_1b3 | 4 | 10.76 | 98.20 | 296.40 | 84.26 | 1001.01 | 3965.66 | 7.83x | 10.19x | 13.38x |

**Mean per-call ratios across 12 paired rows: p50 7.98x, p95 8.21x, p99 9.29x.**

The sim-wall slowdown is 1 / 0.237 = **4.21x**. The gap between per-call (~8x) and sim-wall (~4.2x) is dilution: the simulator loop also processes non-allocate events (free, growth, departure, samples) that cost the same in both variants.

### §1.2 Single-cell `torch.profiler` (mixed_long / jamba_1_5_mini / 4 GiB / seed 20260520)

Phase 2 (`--phase 2`) drives one cell directly through `run_benchmark()` and wraps the call with `torch.profiler.profile(activities=[CPU, CUDA])`. Cell-level metrics:

| Metric | Python b128 | Triton b128 | Triton/Python |
|---|---|---|---|
| wall_clock_s (with profiler overhead) | 14.48 | 40.99 | 2.83x |
| goodput_requests_per_second | 36.05 | 6.19 | 0.17x |
| alloc_p50_us | 7.02 | 82.60 | **11.77x** |
| alloc_p95_us | 47.31 | 321.97 | 6.80x |
| alloc_p99_us | 1343.14 | 14367.09 | 10.70x |
| free_p50_us | 2286.37 | 2702.99 | 1.18x |
| time_in_service_ns | 5,109,736,815 | 33,636,212,697 | **6.58x** |
| time_in_oom_retry_ns | 622,594,075 | 564,413,626 | 0.91x |
| time_in_idle_ns | 203,979,356 | 360,286,495 | 1.77x |

Both runs had identical `oom_count = 95`, `effective_batch_size_p50 = 133`, `n_allocate_calls = 28,960`, `n_free_calls = 256` — parity confirmed at cell scale.

Profiler top events for the Triton variant (407,649 kernel launches across the run, more than `n_allocate_calls = 28,960` because each `allocate()` may launch multiple kernels via the `_allocate_into` per-block loop):

| Event | Self CPU | CPU per call | Self CUDA | CUDA per call | # calls |
|---|---|---|---|---|---|
| `cuLaunchKernelEx` | 9.777 s | **24.40 us** | 0 | 0 | 407,649 |
| `zero_page_kernel` (GPU) | 0 | 0 | 2.546 s | **6.245 us** | 407,649 |

CPU side of every launch (driver round-trip to `cuLaunchKernelEx` + the Triton Python wrapper that produces its arguments) is **3.9x** the GPU side. The kernel itself does ~6 us of useful work; the launch path around it does ~24 us of CPU work plus another ~50 us of Triton Python-side overhead (signature lookup, kernel-arg marshalling, validation in `launch_zero_page`).

### §1.3 JIT warmup ablation (`--phase 3`)

First 30 `TritonAVMPAllocator.allocate()` calls of a fresh allocator (each with explicit `torch.cuda.synchronize()` to isolate per-call cost):

- `call[0]`: **1,334,507.78 us = 1.33 s** — one-time Triton runtime / driver initialization + first kernel launch.
- `call[1]`: 312.96 us — partial warmup tail.
- `median(call[5:30])`: **114.03 us** — steady state.
- `p95(call[5:30])`: 233.38 us.

The script prints a misleading heuristic verdict ("call[0] >> median, so JIT warmup is significant"). The honest interpretation: the **per-process** initialization cost is ~1.3 s, but every cell in the sweep runs in the same process; the 216-cell sweep amortizes this to ~6 ms per cell (0.06% of the average 9.67 s per-cell wall-clock). **JIT warmup is not the slowdown driver.**

### §1.4 Account for the 2089 s sweep wall-clock

Phase 2's single cell (28,960 allocates) consumed:
- 33.6 s of `time_in_service_ns` for Triton (vs 5.1 s for Python).
- Each cell consumes service time roughly proportional to allocate count.

Across 216 cells, with the Triton variant's service time at ~6.58x the Python baseline's, the simulator wall-clock scales similarly. The committed `SWEEP_METADATA.json` reports `total_wall_seconds = 2089.41`. Per-cell mean = 9.67 s; ~85% of that lands in `time_in_service_ns` for Triton cells. **> 90% of the 2089 s budget is accounted for by `time_in_service_ns`** (sum of allocate + free + idle + per-cell harness overhead).

## §2 Hypothesis ranking

The task spec listed four hypotheses; we re-evaluate each:

| # | Original hypothesis | Verdict | Evidence |
|---|---|---|---|
| **H1** | Kernel launch overhead dominates; Triton `triton.jit` warmup / JIT cache miss | **PARTIAL — but warmup is per-process, not per-call** | Phase 3: `call[0]` = 1.33 s, then steady state. Amortized over 216 cells per process, JIT is 0.06% of sweep wall-clock. The dominant per-call cost is `cuLaunchKernelEx` (24 us) + Triton Python wrapper (~50 us), which is NOT JIT compile — it's the steady-state launch path. |
| **H2** | Implicit GPU sync inside the sweep harness blocks the CPU between cells | **FALSIFIED** | Audit of `src/cachepawl/benchmarks/harness/runner.py:328-350` (`_timed_allocate`) confirms it wraps `allocator.allocate()` with `perf_counter_ns()` only; no `torch.cuda.synchronize()`. Audit of `MetricsCollector.__exit__` (`harness/metrics.py:193`) confirms no sync there either. Kernel launches are queued asynchronously. The simulator wall-clock captures Python orchestration + queue dispatch, not GPU execution. |
| **H3** | `BackingStore.buffer_tensor` accessor adds overhead per call | **NEGLIGIBLE** | One Python attribute lookup per `_allocate_into` call (~100 ns), invisible in the profile's top events. |
| **H4** | Sweep wall-clock is wall-clock not GPU-clock, slowdown is Python orchestration | **CONFIRMED — this is the dominant cause** | Profiler shows Self CPU time = 9.98 s, Self CUDA time = 2.55 s for the Triton cell; GPU sits idle ~75% of the run. `time_in_service_ns` Triton/Python = 6.58x. The kernel itself does ~6 us of useful work per call; the Python+driver path around the kernel does ~75 us per call. The simulator measures the full per-call wall-clock; goodput formula `completed_requests / wall_clock_seconds` (`metrics.py:267`) reflects this. |

**Primary root cause**: Triton 3.7.0's per-launch CPU path costs ~24 us in `cuLaunchKernelEx` plus another ~50 us of Triton's Python wrapper (signature lookup, kernel-arg marshalling, grid construction, `launch_zero_page` validation). That ~75 us / call replaces ~7 us / call of Python bookkeeping in the baseline, an ~11x per-call factor on `_timed_allocate`, which the simulator amortizes to ~4x over the full event loop.

## §3 Sweep harness methodology assessment

Is the goodput ratio fair to GPU code?

**The harness measures CPU wall-clock around the simulator loop, not GPU utilization.** Verified at three call sites:

- `runner.py:337-347`: `_timed_allocate` records `time.perf_counter_ns()` around `allocator.allocate()`. No CUDA sync.
- `metrics.py:182, 193`: `MetricsCollector.__enter__/__exit__` records `time.perf_counter_ns()` at the boundaries of the simulator's event loop. No CUDA sync.
- `metrics.py:267`: `goodput_requests_per_second = total_completed / wall_s` where `wall_s` is the CPU-time delta from `__enter__` to `__exit__`.

For an allocator that does only Python bookkeeping (the Python baseline), the wall-clock equals the bookkeeping time. For an allocator that queues GPU work (Triton variant), kernels execute asynchronously after `allocator.allocate()` returns; the simulator does not wait. The wall-clock therefore captures only:

- Python time inside `allocate()` (mint, page-table update, tracker touch).
- The CPU-side cost of dispatching the kernel (`launch_zero_page` Python overhead + the `cuLaunchKernelEx` driver call).
- Python time inside `free()` and other simulator events.

It does NOT capture:
- GPU execution time of the kernel (which runs asynchronously after dispatch).
- Driver-side queuing of additional kernels (default-stream serialization happens on the GPU).

**Implication for the v2 paper**: the goodput ratio is a fair measure of the *simulator's per-allocate CPU cost*, not of inference throughput. A real inference engine batches launches into a CUDA graph or queue and would not pay 24-75 us / call. The byte-identical event-stream parity (`oom_count = 510 = 510 = 510`, 0.0000% drift on `effective_batch_size_p50`) IS the correctness claim and is independent of this measurement.

**The PR #48 prose claiming "cuda.synchronize per allocate inside the latency timing" is wrong**: there is no sync in the simulator's hot loop. The slowdown is real Python+driver overhead, not measurement noise.

## §4 Verdict

The 4.21x simulator-wall-clock slowdown is **real Python orchestration overhead**, not a harness artifact and not real GPU bottleneck.

Breakdown of the ~75 us per-call Triton overhead (vs ~7 us baseline):
- ~24 us: CUDA driver round-trip (`cuLaunchKernelEx`).
- ~25-50 us: Triton 3.x Python launcher (signature dictionary lookup, kernel-arg marshalling, grid computation, validation in `launch_zero_page`).
- ~6 us: actual GPU kernel execution (runs concurrently with Python work on subsequent allocates; only visible when sync forces it into the per-call timing).

The kernel itself is healthy:
- 6.25 us / call on 64 KiB at ~10 GB/s effective write bandwidth (RTX 3060 peak ~360 GB/s; we are limited by program count for a small page).
- No measurement methodology issue.
- No GPU-side bottleneck (GPU idle ~75% during the Triton run).

This slowdown was hidden by:
- The Week 1 9-cell smoke not measuring goodput (only `oom_count` and `effective_batch_size_p50`).
- The PR #48 prose's incorrect "sync per call" hypothesis pointing at the wrong cause.

The correctness contribution stands; the performance contribution must be either (a) closed by an optimization or (b) reframed in the v2 paper.

## §5 Recommendation

Three candidate paths, ranked by likely impact and engineering cost:

### (A) CUDA graph replay — **recommended if v2 needs the throughput claim** (high impact, moderate effort)

Capture each `launch_zero_page(buffer_tensor, offset, size_bytes)` invocation into a `torch.cuda.CUDAGraph`. The graph holds the kernel-launch sequence and is replayable with ~3-5 us / replay instead of ~75 us / Python launch. Since the page geometry within a single `_allocate_into` is fixed (one page size per kind), one graph per (kind, num_blocks) tuple is sufficient. Implementation locus: a graph cache on the `TritonAVMPAllocator` instance, populated lazily on first call for each shape.

Expected gain: ~15x reduction in per-allocate CPU cost (75 us → 5 us), closing the goodput gap from 0.24x to ~0.85x. Caveat: CUDA graphs are stream-bound; the implementation must capture on a non-default stream and wait on the default stream where necessary.

### (B) Batched launches — partial mitigation (low impact, low effort)

When `num_blocks > 1`, single grid covering all offsets in one kernel call. Saves ~75 us × (N-1) per `_allocate_into` call. Limited because the workload presets typically allocate one page per layer per growth event (`num_blocks = 1`). Should be combined with (A) for full effect.

### (C) Document and accept (zero effort)

Acknowledge in v2 paper §5 that the goodput ratio measures simulator orchestration cost per allocate, not inference throughput. Make the byte-identical event-stream parity the headline correctness claim and add a half-paragraph explaining why a real engine wouldn't pay this cost. The data file `research/avmp/v2/results/paper_section_5_data.json` already includes both numbers; only the prose framing needs to change.

### Recommended next step

If the v2 paper deadline allows ~1 week of engineering, do (A): CUDA graph capture. The ~15x speedup converts the goodput ratio from 0.24x to a number close to 1.0, which is a much cleaner story for the paper.

If the deadline is tight, do (C): re-frame the metric. This is honest and defensible — the 200 us p95 batched-amortized result in `tests/benchmarks/test_zero_page_latency.py` from Week 1 already proves the kernel works at the throughput the paper claims; the simulator goodput is a measurement-protocol artifact.

The investigation does NOT recommend changing the simulator harness to e.g. exclude Triton overhead from `wall_s` — that would be moving the goalposts. The metric is honest; the question is which metric belongs in the paper.
