# Triton Roadmap: AVMP v2 Hardware Realization

Three-week plan to land a Triton-backed `TritonAVMPAllocator` that realizes
the Python AVMP prototype on real RTX 3060 hardware. Target venue: ML for
Systems @ NeurIPS 2026 workshop.

Audit prerequisites in `research/avmp/v2/TRITON_AUDIT.md`. Read that first;
this file references its findings rather than restating them.

## §1 Scope

### New Triton kernels (no existing GPU path is being ported)

- `zero_page_kernel` (`src/cachepawl/kernels/allocate.py`): zero-fill a
  region of `BackingStore._buffer` matching one freshly allocated KV page
  or SSM block. Required because the Python prototype hands back
  uninitialized memory (the simulation never reads it); a real engine
  needs zeros.
- `copy_region_kernel` (`src/cachepawl/kernels/migrate.py`): physical
  byte copy used during migration when an outstanding handle's physical
  offset must be relocated. **Deferred to v2.1** if `_apply_rebalance`'s
  current "shrink only the active counter, leave live offsets untouched"
  semantic carries over to the cuMemMap-backed pool. The kernel is
  scaffolded now to keep the roadmap honest about what could fail.

### Stays Python

- `VirtualPageTable` (mint / resolve / remove) — pure dict bookkeeping;
  GIL is not the bottleneck here, kernel launch overhead would be.
- `PoolPressureMonitor` and `PoolPressureState` — state machine
  transitions are not on the data path; one branch per allocate.
- `_apply_rebalance` orchestration — sequencing of donor shrink /
  recipient grow stays Python; kernel launches happen *inside* the
  resize_capacity path if `copy_region_kernel` is needed.
- `_try_bulk_allocate_with_eviction` and the `CapacityError` retry
  ladder (RFC 0002 §4.2).
- All stats accounting in `get_allocator_stats()`.

### Class layout decision

Preferred: `TritonAVMPAllocator(AsymmetricVirtualPool)` subclass in
`src/cachepawl/allocator/avmp/triton_allocator.py`. Only methods that
gain kernel launches are overridden. The state machine, capacity-error
retry, observability surface, and `get_allocator_stats()` come for free
from the base class.

Fallback (used only if subclassing the v1 pool turns out to require too
many overrides — e.g., if the `KVPagesStore` / `SSMBlocksStore`
construction in `AsymmetricVirtualPool.__init__` cannot be swapped for
cuMemMap-backed stores without restructuring): sibling package
`src/cachepawl/allocator/avmp_triton/` that *imports* the shared helpers
(`VirtualPageTable`, `PoolPressureMonitor`, `_apply_rebalance` extracted
to a module-level function) rather than re-implementing them. Decision
recorded here once Week 1 lands.

### Performance hypothesis

Per-`allocate()` cost decomposition on RTX 3060 (CC 8.6):

| Component | Estimate |
|---|---|
| Python bookkeeping (`mint`, dict updates, monitor tick) | ~10 µs |
| Triton kernel launch overhead (one launch per page) | ~10-30 µs |
| Kernel execution (zero-fill ~64 KiB at ~400 GB/s peak BW) | ~0.5 µs |
| `torch.cuda.Event` sync overhead in measurement | ~1 µs |

Kernel launch dominates. Batching multiple page zero-fills into one
launch is the obvious mitigation if the per-call budget is missed.

## §2 Kernel API design

### `zero_page_kernel`

```python
@triton.jit
def zero_page_kernel(
    buffer_ptr,                       # *u8: BackingStore._buffer pointer
    offset: int,                      # byte offset into the buffer
    size_bytes: int,                  # number of bytes to zero
    BLOCK_SIZE: tl.constexpr,         # bytes per program; compile-time
) -> None:
    pid = tl.program_id(axis=0)
    block_start = offset + pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < (offset + size_bytes)
    tl.store(buffer_ptr + offs, tl.zeros((BLOCK_SIZE,), dtype=tl.uint8), mask=mask)
```

- **Grid**: `(triton.cdiv(size_bytes, BLOCK_SIZE),)`. One program per
  `BLOCK_SIZE` chunk.
- **`BLOCK_SIZE`**: 4096 bytes is the planned default — 4 KiB matches the
  driver's preferred coalesced-store width and keeps register pressure
  trivial.
- **Shared memory**: zero; this is a pure streaming store.
- **Coalescing**: thread `i` writes byte `i` of the block, so threads in
  a warp write contiguous bytes — fully coalesced.
- **Num warps / stages**: default `num_warps=4`, `num_stages=2`. Will
  re-tune in Week 1 after the first micro-bench.

### `copy_region_kernel` (deferred to v2.1 if not needed)

```python
@triton.jit
def copy_region_kernel(
    src_ptr,                          # *u8
    dst_ptr,                          # *u8
    num_bytes: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_bytes
    vals = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, vals, mask=mask)
```

- Same grid / coalescing / config as `zero_page_kernel`.
- Used only if migration needs to physically relocate live blocks. RFC
  0002 §4.5 calls out the write barrier semantics required between
  donor shrink and recipient grow when this kernel is involved.

### Hardware envelope (RTX 3060, CC 8.6)

- 28 SMs, max 1024 threads/block, 49 KiB shared mem/block, 65K
  registers/SM.
- Memory bandwidth ~360 GB/s effective.
- AVMP's 2× physical-footprint cost (RFC 0002 §4.3) limits useful pool
  size to ~4 GiB on this 12 GiB card unless cuMemMap is wired up to
  eliminate the duplicate buffer.

## §3 Validation plan

### Parity smoke (9 cells)

Run the existing benchmark harness with both `avmp_dynamic` (Python) and
the new `TritonAVMPAllocator` on:

- **3 workloads**: `uniform_short`, `mixed_long`, `agentic_burst`.
- **1 model spec**: the smallest available (TBD when configs land).
- **1 total_bytes**: 4 GiB (the realistic ceiling on RTX 3060 given the
  2× footprint).
- **3 seeds**: `[0, 1, 2]`.

Total: 3 × 1 × 1 × 3 = **9 cells**. Multi-seed is deliberate; flaky
kernel races (under-synchronized `cuda.Event` framing, missing
`cuda.synchronize()` between launches) typically surface as
seed-dependent OOM-count drift. A 1-cell smoke would hide them.

Pass criteria, per cell:

- `total_oom`: identical to within ±1 event.
- `effective_batch_size_p50`: within 1% of the Python baseline.
- `fragmentation_p50`: within 1% of the Python baseline.

### Full validation (270 cells)

The full validation rerun uses the existing
`THRESHOLD_SWEEP_STAGE2_VARIANTS` sweep in
`src/cachepawl/benchmarks/compare/sweep.py:677-702`:
5 variants × 3 workloads × 2 specs × 3 byte-sizes × 3 seeds = 270 cells.
We swap one of the variants for the Triton-backed allocator and re-run on
RTX 3060 once the 9-cell smoke passes.

### Latency target: ≤ 200 µs per `allocate()` (p50)

Measurement protocol (mandatory; the Week 1 commit ships the harness):

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
allocator.allocate(num_blocks, dtype_bytes=dtype_bytes)
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

- Record N = 10,000 iterations; report `p50`, `p99`.
- Warm up with 100 untimed iterations to prime the Triton autotune cache.
- Separate timers for `allocate()`, `free()`, `migrate_capacity()`.

Budget rationale: 200 µs/call ≈ 5,000 allocate/s sustained. The 270-cell
sweep at typical workload size has ~10⁶ allocate calls per cell across
all cells, so the sweep would run in ~200 s of pure allocator time —
small relative to the simulation's other costs. Above 200 µs, the
allocator becomes the bottleneck and the paper's claim that "the
allocator's overhead is invisible at sustained inference throughput" is
in danger.

If the kernel-launch hypothesis from §1 holds (10-30 µs/launch), 200 µs
leaves headroom for the Python bookkeeping and one or two kernel
launches per call.

## §4 Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `cuMemMap` not available on dev box (driver / CUDA version) | medium | Verify CUDA ≥ 11.2 and `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` support before committing to that path. Fallback: pre-allocated buffer matches the Python prototype, only the 2× footprint constraint stays. |
| Triton 3.x API drift from documentation | low | Pin `triton>=3.0` (already in `pyproject.toml`). Use `tl.constexpr` and `triton.cdiv` per the 3.x conventions. Pre-check the kernel signatures against the installed Triton version on Week 1 day 1. |
| RTX 3060 12 GiB cap with 2× footprint | high | Already known; cap useful pool at 4 GiB via `--max-total-bytes 4294967296`. cuMemMap removes this risk if wired up. |
| Shared-memory pressure on large block migrations | low | `copy_region_kernel` does not use shared memory (streaming load/store). If a future tile-based variant uses shared memory, cap tile size at 8 KiB to stay well under 49 KiB. |
| Kernel-launch overhead exceeds 200 µs budget | medium | Batch multiple page zero-fills into one launch (group by contiguous offsets). Re-tune `BLOCK_SIZE`, `num_warps`, `num_stages`. Fallback: relax target to 500 µs and document the tradeoff in §3 of the paper. |
| Flaky kernel races invisible at seed=0 | medium | 9-cell smoke runs 3 seeds; full sweep runs 3 seeds. CI runs the unit tests with `torch.cuda.synchronize()` after every kernel call to surface ordering bugs. |

## §5 Milestones (3 weeks)

| Week | Deliverable | Definition of done |
|---|---|---|
| 1 | `zero_page_kernel` body + unit tests + `TritonAVMPAllocator._allocate_into` override that calls the kernel | Single-cell parity test passes (`avmp_dynamic` vs `TritonAVMPAllocator` on 1 workload, 1 seed). `torch.cuda.Event` micro-bench reports p50 ≤ 200 µs. |
| 2 | `copy_region_kernel` body OR explicit "not needed" decision + state-machine integration (override `_apply_rebalance` if kernel is needed) | 9-cell smoke parity passes. Decision on subclass-vs-sibling fallback recorded in §1. |
| 3 | Full 270-cell sweep on RTX 3060 + paper §5 (Hardware Realization) draft | Sweep artifacts committed under `benchmarks/results/avmp-v2-triton/`. Paper section drafted with the latency p50/p99 numbers and the parity table. |

PR cadence: one PR per week, each ≤ 400 LOC net (per global git rules).
The current PR (scaffold + docs) is the prerequisite for Week 1.
