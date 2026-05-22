"""Microbench probe: CUDA graph replay variants for AVMP `zero_page_kernel`.

PR #49 (slowdown root cause) showed the Triton allocate path pays ~75 us
CPU per call vs ~7 us for the Python baseline. PR #49 section 5 listed
three options to close the gap; option (A) was CUDA graph replay,
expected to reduce per-call overhead to ~5 us by eliminating the
Triton Python launcher and the cuLaunchKernelEx driver round-trip.

This script measures whether (A) actually achieves that. It compares
five variants at the same 64 KiB page size and `BLOCK_SIZE = 1024` used
by the Week 1 latency benchmark, on RTX 3060 / CUDA 13.0 / Triton 3.7.0:

- V1: ``device_tensor[0] = N`` + direct kernel launch.
- V2: pinned-host buffer -> ``copy_(non_blocking=True)`` + direct kernel.
- V3: graph replay where the pinned-host -> device ``copy_`` is captured
  inside the graph; pre-write `host_offset[0] = N` per call, then
  ``g.replay()``. This is the design implied by PR #49 option (A).
- V3-sync: V3 with an explicit ``torch.cuda.synchronize()`` per call.
- V4: graph captured with a static offset, replayed many times. This
  measures the irreducible ``g.replay()`` floor.

Findings landed in ``research/avmp/v2/GRAPH_REPLAY_FEASIBILITY.md``.
The script exists in committed form so future runs can validate the
finding or detect regressions if a Triton / PyTorch upgrade changes
the per-call cost surface.

Usage:

    uv run python research/avmp/v2/results/probe_graph_replay.py

Expected runtime: ~30 s on RTX 3060. The kernel runs ~25K times across
the five variants; warmup is a single 20-iter pre-loop at the top.
"""

from __future__ import annotations

import statistics
import time

import torch
import triton
import triton.language as tl


# The kernel: same shape as `cachepawl.kernels.zero_page_kernel` but with
# `offset` loaded from a device tensor pointer so the captured graph can
# observe an offset that varies per replay via a pinned-host pre-write.
# Single ignore on the @triton.jit decorator silences the mypy strict
# untyped-decorator complaint; the inner def is hidden behind the
# decorator's Any return.
@triton.jit  # type: ignore[untyped-decorator]
def zero_with_offset_ptr(  # type: ignore[no-untyped-def]
    buffer_ptr,
    offset_ptr,
    size_bytes,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(axis=0)
    offset = tl.load(offset_ptr)
    block_start = offset + pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < (offset + size_bytes)
    tl.store(buffer_ptr + offs, tl.zeros((BLOCK_SIZE,), dtype=tl.uint8), mask=mask)


_BLOCK = 1024
_SIZE = 65536  # 64 KiB; matches Week 1 KV page on jamba_1_5_mini
_N = 5000
_P95_IDX = int(0.95 * _N)


def _summarize(label: str, samples: list[float]) -> None:
    samples.sort()
    p50 = statistics.median(samples)
    p95 = samples[_P95_IDX]
    print(f"{label:60s}  p50 {p50:7.2f} us, p95 {p95:7.2f} us")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this probe requires an RTX 3060 or similar")

    buf = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    device_offset = torch.zeros(1, dtype=torch.int64, device="cuda")
    grid = (triton.cdiv(_SIZE, _BLOCK),)

    # Warmup: prime the Triton JIT cache and the CUDA context.
    for _ in range(20):
        zero_with_offset_ptr[grid](buf, device_offset, _SIZE, BLOCK_SIZE=_BLOCK)
    torch.cuda.synchronize()

    # V1: device_tensor[0] = N + direct kernel launch.
    samples: list[float] = []
    for i in range(_N):
        t0 = time.perf_counter_ns()
        device_offset[0] = i * _BLOCK
        zero_with_offset_ptr[grid](buf, device_offset, _SIZE, BLOCK_SIZE=_BLOCK)
        samples.append((time.perf_counter_ns() - t0) / 1000)
    _summarize("V1 tensor[0]=N + direct kernel (no sync)", samples)

    # V2: pinned-host buffer -> copy_(non_blocking=True) + direct kernel.
    host_offset = torch.empty(1, dtype=torch.int64, pin_memory=True)
    samples = []
    for i in range(_N):
        t0 = time.perf_counter_ns()
        host_offset[0] = i * _BLOCK
        device_offset.copy_(host_offset, non_blocking=True)
        zero_with_offset_ptr[grid](buf, device_offset, _SIZE, BLOCK_SIZE=_BLOCK)
        samples.append((time.perf_counter_ns() - t0) / 1000)
    _summarize("V2 pinned host->copy_ + direct kernel (no sync)", samples)

    # V3: capture copy_ + kernel into a CUDA graph; pre-write host_offset
    # before each replay. This is the PR #49 (A) design.
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.current_stream())
        for _ in range(5):
            device_offset.copy_(host_offset, non_blocking=True)
            zero_with_offset_ptr[grid](buf, device_offset, _SIZE, BLOCK_SIZE=_BLOCK)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream), torch.cuda.graph(g, stream=stream):
        device_offset.copy_(host_offset, non_blocking=True)
        zero_with_offset_ptr[grid](buf, device_offset, _SIZE, BLOCK_SIZE=_BLOCK)

    samples = []
    for i in range(_N):
        t0 = time.perf_counter_ns()
        host_offset[0] = i * _BLOCK
        g.replay()
        samples.append((time.perf_counter_ns() - t0) / 1000)
    _summarize("V3 graph replay (pinned pre-write, no sync)", samples)

    samples = []
    for i in range(_N):
        t0 = time.perf_counter_ns()
        host_offset[0] = i * _BLOCK
        g.replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1000)
    _summarize("V3 graph replay (pinned pre-write + sync per call)", samples)

    # V4: static-offset graph capture; offset is baked in at capture
    # time, no pre-write needed. Measures the irreducible replay floor.
    g_static = torch.cuda.CUDAGraph()
    dev_off_static = torch.tensor([4096], dtype=torch.int64, device="cuda")
    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.current_stream())
        for _ in range(5):
            zero_with_offset_ptr[grid](buf, dev_off_static, _SIZE, BLOCK_SIZE=_BLOCK)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream), torch.cuda.graph(g_static, stream=stream):
        zero_with_offset_ptr[grid](buf, dev_off_static, _SIZE, BLOCK_SIZE=_BLOCK)

    samples = []
    for _ in range(_N):
        t0 = time.perf_counter_ns()
        g_static.replay()
        samples.append((time.perf_counter_ns() - t0) / 1000)
    _summarize("V4 static-offset graph replay only (no sync, floor)", samples)

    print()
    print("Week 1 baseline (PR #49 report): ~75 us/call (no sync, in the simulator)")
    print("Gate (PR #49 option A premise): <=15 us/call to claim >=5x speedup")


if __name__ == "__main__":
    main()
