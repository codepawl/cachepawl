# §5 Hardware Realization — draft for v2 paper

Length target: ~0.5 page in NeurIPS workshop format (4-page extended-abstract
limit). The committed data backing every number lives in
`research/avmp/v2/results/paper_section_5_data.json`, and the
sweep artifacts under `research/avmp/v2/results/sweep-triton-validation/`.
Source documents for the cost-surface analysis: PR #47 (Week 1 kernel),
PR #48 (sweep parity), PR #49 (slowdown root cause), PR #50 (graph replay
feasibility). Markdown for now; LaTeX conversion lands when the v2 paper
draft tree is created.

## Implementation

We implemented `TritonAVMPAllocator`, a hardware-backed variant of the
Python AVMP allocator that performs byte-level page zero-fill via a
Triton kernel (`zero_page_kernel`: 1-D scatter, `BLOCK_SIZE = 1024`,
fully coalesced with a tail mask). The allocator subclasses the Python
prototype (`AsymmetricVirtualPool`), overriding only the per-page
allocation hook (`_allocate_into`) while preserving the state machine,
dynamic rebalancing, capacity-error retry, and accounting paths
unchanged.

## Parity validation

Across the full 216-cell paired sweep (3 workloads × 2 model specs × 2
pool budgets × 3 seeds = 216 cells; 12 paired aggregated rows = 36
paired seed-cells for the head-to-head `avmp_dynamic_b128_triton` vs
`avmp_dynamic_b128`), the Triton implementation achieves byte-identical
event-stream parity with the Python prototype:

| Metric | Result |
|---|---|
| OOM count drift across 36 paired seed-cells | **0.00 %** |
| `effective_batch_size_p50` max relative drift | **0.00 %** |
| Migration count and `bytes_migrated_total` | byte-identical |
| Cross-workload OOM aggregate (Triton / Python / v1 reference) | **510.00 / 510.00 / 510.00** |

This confirms AVMP's design holds on real GPU memory and the byte-level
zero-fill kernel preserves the allocator's correctness invariants.

## Performance characterization

The Triton sweep wall-clock is **2089 s** versus **974 s** for Python
(2.1× slower). The cost is not GPU-bound: per-`allocate()` decomposition
locates ~75 µs of Python+driver work per call for Triton vs ~7 µs for
Python (PR #49). The breakdown of the Triton per-call cost: ~24 µs
`cuLaunchKernelEx` driver round-trip + ~50 µs Triton Python launcher
(signature lookup, kernel-arg marshalling, grid construction) + ~6 µs
actual GPU kernel execution. The GPU is active for only ~25 % of the
sweep wall-clock.

## Graph replay attempt and PyTorch API limitation

We tested CUDA graph replay (Strategy A: capture the kernel launch into
a `torch.cuda.CUDAGraph`, replay against a pinned-host offset tensor)
on the same hardware (RTX 3060, CUDA 13.0, PyTorch 2.12.0+cu130, Triton
3.7.0). The captured-replay floor (static offset baked in at capture) is
7 µs per call, meeting the 5× speedup target — but requires one
captured graph per unique offset value, which is ~650 MiB of graph
metadata for a 4 GiB pool with 64 KiB pages: infeasible. The
varying-offset variant (pre-write `host_offset[0] = N`, replay) measures
**53.80 µs per call** — only ~1.4× faster than Week 1, 3.6× above the
target. PyTorch 2.12.0 does not expose `cudaGraphExecKernelNodeSetParams`,
the CUDA primitive that would let us update captured-kernel arguments at
replay time without a tensor write. Strategy A is not viable until either
PyTorch exposes this API or the allocator adopts a batched/deferred
launch surface; see `research/avmp/v2/SLOWDOWN_ROOT_CAUSE.md` and
`research/avmp/v2/GRAPH_REPLAY_FEASIBILITY.md` for the full analysis.

## Implications

The Triton implementation serves as a **correctness oracle** validating
AVMP's design on real GPU memory. Production deployment of the
per-allocate kernel requires either (a) a batched/deferred allocation
API with an explicit per-decode-step `flush()` boundary that amortizes
the Python+driver overhead across many allocations, or (b) updates to
the PyTorch CUDA graph API supporting kernel-parameter rebinding
without recapture. We defer the batched API design to v2.1.

For the throughput evaluation in §4, we use the Python allocator path:
the Triton implementation establishes correctness, not the performance
baseline. The v2 paper's core performance contribution is the vLLM
integration described in §6.
