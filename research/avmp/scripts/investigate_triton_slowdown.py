"""Investigate the 4x simulator-wall-clock slowdown of TritonAVMPAllocator.

Three phases, each callable individually via --phase:

  Phase 1 — Decompose from existing sweep data (no GPU). Reads
  ``research/avmp/v2/results/sweep-triton-validation/aggregated.json``
  and tabulates per-call `allocate_*_ns` ratios between
  `avmp_dynamic_b128` (Python) and `avmp_dynamic_b128_triton` for the
  12 paired aggregated rows.

  Phase 2 — Single-cell profile with `torch.profiler`. Drives one
  representative cell (mixed_long / jamba_1_5_mini / 4 GiB / seed
  20260520) through ``run_benchmark`` for both the Python baseline
  and Triton variant, with the profiler wrapped around the call.
  Saves Chrome traces and prints the top events tables. Also reads
  the BenchmarkRun's phase-time decomposition fields (`time_in_*_ns`,
  schema 1.3.0) to decompose simulator wall-clock without parsing
  the trace JSON.

  Phase 3 — JIT warmup ablation. Records per-call latencies for the
  first 20 ``TritonAVMPAllocator.allocate()`` calls of a fresh
  allocator. Compares call[0] (potential JIT cost) to the median of
  call[5:] (steady state). Ratios > 1.5x flag JIT compile as
  significant.

The investigation is read-only: no allocator or kernel code is
changed. Verdict + recommendation land in
``research/avmp/v2/SLOWDOWN_ROOT_CAUSE.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.profiler

from cachepawl.allocator.avmp import AsymmetricVirtualPool, TritonAVMPAllocator
from cachepawl.benchmarks import (
    PRESETS,
    HybridModelSpec,
    LayerKind,
    LayerSpec,
    WorkloadSpec,
    run_benchmark,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGG_JSON = _REPO_ROOT / "research/avmp/v2/results/sweep-triton-validation/aggregated.json"

_TOTAL_BYTES = 4 * 1024**3
_ROOT_SEED = 20260520
_WORKLOAD_NAME = "mixed_long"


def _hybrid_spec_from_workload(spec: WorkloadSpec) -> HybridModelSpec:
    """Mirror ``cachepawl.benchmarks._hybrid_spec_from_workload``."""

    total_layers = spec.attention_layers + spec.ssm_layers
    if spec.attention_layers <= 0:
        attn_every = total_layers + 1
    else:
        attn_every = max(1, total_layers // spec.attention_layers)
    layers = tuple(
        LayerSpec(
            index=i,
            kind=LayerKind.ATTENTION if i % attn_every == 0 else LayerKind.MAMBA2,
        )
        for i in range(total_layers)
    )
    ratio = float(spec.attention_layers) / float(max(1, spec.ssm_layers))
    return HybridModelSpec(
        name=f"workload-{spec.name}",
        layers=layers,
        attention_to_ssm_ratio=ratio,
        attention_profile=spec.attention_profile,
        ssm_profile=spec.ssm_profile,
        dtype=spec.dtype,
    )


def _build_python_allocator(workload: WorkloadSpec, device: torch.device) -> AsymmetricVirtualPool:
    return AsymmetricVirtualPool(
        model_spec=_hybrid_spec_from_workload(workload),
        total_bytes=_TOTAL_BYTES,
        device=device,
        mamba_ratio=0.5,
        migration_batch_size=128,
        rebalance_enabled=True,
    )


def _build_triton_allocator(workload: WorkloadSpec, device: torch.device) -> TritonAVMPAllocator:
    return TritonAVMPAllocator(
        model_spec=_hybrid_spec_from_workload(workload),
        total_bytes=_TOTAL_BYTES,
        device=device,
        mamba_ratio=0.5,
        migration_batch_size=128,
        rebalance_enabled=True,
    )


def _us(ns: int | float) -> str:
    return f"{ns / 1000:.2f}"


def phase1() -> None:
    """Phase 1: aggregated.json decomposition table (CPU only)."""

    if not _AGG_JSON.exists():
        print(f"phase1: missing {_AGG_JSON}", file=sys.stderr)
        sys.exit(1)

    rows = json.loads(_AGG_JSON.read_text())["rows"]
    py_rows = {
        (r["workload_name"], r["model_spec_name"], r["total_bytes"]): r
        for r in rows
        if r["variant_label"] == "avmp_dynamic_b128"
    }
    tr_rows = {
        (r["workload_name"], r["model_spec_name"], r["total_bytes"]): r
        for r in rows
        if r["variant_label"] == "avmp_dynamic_b128_triton"
    }

    print("## Phase 1 — per-call allocate latency from aggregated.json")
    print()
    print(
        "(median over 3 seed replicates per cell; `allocate_*_ns_median` field of "
        "`aggregated.json`)"
    )
    print()
    print(
        "| workload | spec | pool_GiB | py p50 us | py p95 us | py p99 us | "
        "tr p50 us | tr p95 us | tr p99 us | p50 ratio | p95 ratio | p99 ratio |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")

    p50_ratios: list[float] = []
    p95_ratios: list[float] = []
    p99_ratios: list[float] = []
    for key in sorted(py_rows.keys() & tr_rows.keys()):
        py = py_rows[key]
        tr = tr_rows[key]
        py_p50 = py["allocate_p50_ns_median"] / 1000
        py_p95 = py["allocate_p95_ns_median"] / 1000
        py_p99 = py["allocate_p99_ns_median"] / 1000
        tr_p50 = tr["allocate_p50_ns_median"] / 1000
        tr_p95 = tr["allocate_p95_ns_median"] / 1000
        tr_p99 = tr["allocate_p99_ns_median"] / 1000
        r50 = tr_p50 / py_p50 if py_p50 > 0 else 0.0
        r95 = tr_p95 / py_p95 if py_p95 > 0 else 0.0
        r99 = tr_p99 / py_p99 if py_p99 > 0 else 0.0
        p50_ratios.append(r50)
        p95_ratios.append(r95)
        p99_ratios.append(r99)
        gib = key[2] // 1024**3
        print(
            f"| {key[0]} | {key[1]} | {gib} | "
            f"{py_p50:.2f} | {py_p95:.2f} | {py_p99:.2f} | "
            f"{tr_p50:.2f} | {tr_p95:.2f} | {tr_p99:.2f} | "
            f"{r50:.2f}x | {r95:.2f}x | {r99:.2f}x |"
        )

    print()
    print(
        f"Mean per-call ratios across {len(p50_ratios)} paired rows: "
        f"p50 {statistics.mean(p50_ratios):.2f}x, "
        f"p95 {statistics.mean(p95_ratios):.2f}x, "
        f"p99 {statistics.mean(p99_ratios):.2f}x"
    )

    # Compare with the overall simulator-wall-clock ratio from paper_section_5_data.json.
    paper_path = _REPO_ROOT / "research/avmp/v2/results/paper_section_5_data.json"
    if paper_path.exists():
        paper = json.loads(paper_path.read_text())
        goodput_ratio = paper["goodput"]["overall"]["ratio_triton_over_python_mean"]
        sim_slowdown = 1.0 / goodput_ratio
        print(f"Overall simulator-wall-clock slowdown (1 / goodput ratio): {sim_slowdown:.2f}x")
        print(
            f"Per-call p50 ratio averages ~{statistics.mean(p50_ratios):.1f}x; "
            f"sim-wall slowdown is ~{sim_slowdown:.1f}x. The gap is expected: "
            "the simulator loop also processes non-allocate events (free, growth, "
            "departure, samples), which cost the same in both variants and dilute "
            "the per-call ratio when amortized over total wall-clock."
        )


def _profile_cell(
    label: str,
    builder: Callable[[WorkloadSpec, torch.device], AsymmetricVirtualPool],
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    """Profile a single cell with torch.profiler; return key stats."""

    workload = dataclasses.replace(PRESETS[_WORKLOAD_NAME], seed=_ROOT_SEED)
    allocator = builder(workload, device)

    trace_path = output_dir / f"profile_{label}_chrome.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n## Phase 2 [{label}]: profiling one cell ({_WORKLOAD_NAME}, "
        f"jamba_1_5_mini, 4 GiB, seed {_ROOT_SEED}) ..."
    )

    t0 = time.perf_counter_ns()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        run = run_benchmark(
            allocator=allocator,
            spec=workload,
            allocator_name=f"investigate-{label}",
            output_dir=output_dir / f"runs-{label}",
            device="cuda",
        )
    wall_s = (time.perf_counter_ns() - t0) / 1e9

    prof.export_chrome_trace(str(trace_path))
    trace_size_mib = trace_path.stat().st_size / (1024 * 1024)
    print(f"  saved Chrome trace: {trace_path.name} ({trace_size_mib:.1f} MiB)")

    print("\n  --- top 15 events by CUDA time total ---")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
    print("\n  --- top 15 events by CPU time total ---")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

    m = run.metrics
    alloc_pct = m.allocate_latency_percentiles()
    n_alloc = len(m.allocate_latency_ns)
    n_free = len(m.free_latency_ns)
    free_pct = m.free_latency_percentiles()

    print(f"\n  --- BenchmarkRun metrics for {label} ---")
    print(f"  wall_clock_s (including profiler overhead): {wall_s:.3f}")
    print(f"  goodput_requests_per_second: {m.goodput_requests_per_second:.4f}")
    print(f"  oom_count: {m.oom_count}")
    print(f"  effective_batch_size_p50: {m.effective_batch_size_p50}")
    print(f"  n_allocate_calls: {n_alloc}")
    print(f"  n_free_calls: {n_free}")
    print(
        f"  allocate_p50/p95/p99 us: "
        f"{alloc_pct.p50_ns / 1000:.2f} / "
        f"{alloc_pct.p95_ns / 1000:.2f} / "
        f"{alloc_pct.p99_ns / 1000:.2f}"
    )
    print(
        f"  free_p50/p95/p99 us: "
        f"{free_pct.p50_ns / 1000:.2f} / "
        f"{free_pct.p95_ns / 1000:.2f} / "
        f"{free_pct.p99_ns / 1000:.2f}"
    )
    print("  --- phase-time decomposition (schema 1.3.0, ns) ---")
    print(
        f"  time_in_service_ns:    {m.time_in_service_ns:>16,}  "
        f"({_us(m.time_in_service_ns):>10} us)"
    )
    print(
        f"  time_in_oom_retry_ns:  {m.time_in_oom_retry_ns:>16,}  "
        f"({_us(m.time_in_oom_retry_ns):>10} us)"
    )
    print(
        f"  time_in_migration_ns:  {m.time_in_migration_ns:>16,}  "
        f"({_us(m.time_in_migration_ns):>10} us; overlaps service)"
    )
    print(f"  time_in_idle_ns:       {m.time_in_idle_ns:>16,}  ({_us(m.time_in_idle_ns):>10} us)")
    total_phase_ns = m.time_in_service_ns + m.time_in_oom_retry_ns + m.time_in_idle_ns
    print(
        f"  phase sum (service+oom_retry+idle): {total_phase_ns:,} ns "
        f"({_us(total_phase_ns):>10} us)"
    )

    return {
        "label": label,
        "wall_clock_s": wall_s,
        "goodput": m.goodput_requests_per_second,
        "n_alloc": n_alloc,
        "n_free": n_free,
        "alloc_p50_us": alloc_pct.p50_ns / 1000,
        "alloc_p95_us": alloc_pct.p95_ns / 1000,
        "alloc_p99_us": alloc_pct.p99_ns / 1000,
        "free_p50_us": free_pct.p50_ns / 1000,
        "free_p95_us": free_pct.p95_ns / 1000,
        "time_in_service_ns": m.time_in_service_ns,
        "time_in_oom_retry_ns": m.time_in_oom_retry_ns,
        "time_in_migration_ns": m.time_in_migration_ns,
        "time_in_idle_ns": m.time_in_idle_ns,
        "trace_path": str(trace_path),
    }


def phase2(profile_output: Path) -> dict[str, dict[str, object]]:
    """Phase 2: torch.profiler around one cell, each allocator."""

    if not torch.cuda.is_available():
        print("phase2: CUDA not available; aborting", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    stats = {}
    stats["python"] = _profile_cell("python", _build_python_allocator, device, profile_output)
    stats["triton"] = _profile_cell("triton", _build_triton_allocator, device, profile_output)
    print("\n## Phase 2 summary table")
    print()
    print("| Metric | Python b128 | Triton b128 | Triton/Python |")
    print("|---|---|---|---|")
    py = stats["python"]
    tr = stats["triton"]

    def fmt_ratio(a: float, b: float) -> str:
        if b == 0:
            return "n/a"
        return f"{a / b:.2f}x"

    for field in (
        "wall_clock_s",
        "goodput",
        "alloc_p50_us",
        "alloc_p95_us",
        "alloc_p99_us",
        "free_p50_us",
        "free_p95_us",
        "time_in_service_ns",
        "time_in_oom_retry_ns",
        "time_in_idle_ns",
    ):
        py_v = py[field]
        tr_v = tr[field]
        py_disp = f"{py_v:,}" if isinstance(py_v, int) else f"{py_v:.4f}"
        tr_disp = f"{tr_v:,}" if isinstance(tr_v, int) else f"{tr_v:.4f}"
        ratio_disp = fmt_ratio(float(tr_v), float(py_v))  # type: ignore[arg-type]
        print(f"| {field} | {py_disp} | {tr_disp} | {ratio_disp} |")

    return stats


def phase3() -> None:
    """Phase 3: first-call vs steady-state ablation for Triton variant."""

    if not torch.cuda.is_available():
        print("phase3: CUDA not available; aborting", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    workload = dataclasses.replace(PRESETS[_WORKLOAD_NAME], seed=_ROOT_SEED)
    allocator = _build_triton_allocator(workload, device)
    allocator.set_current_layer_kind(LayerKind.ATTENTION)
    allocator.set_current_request_id(1)

    print("## Phase 3 — JIT warmup ablation (first-call vs steady-state)")
    print()
    print(
        "Measures per-allocate wall-clock for the first N calls of a fresh "
        "TritonAVMPAllocator. If call[0] >> median(call[5:]), the Triton JIT "
        "warmup is a significant cost; otherwise the per-call cost is "
        "dispatch-only (cache hit on every call from this process onward, "
        "since the on-disk Triton cache is already populated)."
    )
    print()

    n = 30
    times_us: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        ids = allocator.allocate(num_blocks=1, dtype_bytes=2)
        torch.cuda.synchronize()  # to be conservative about steady-state
        elapsed_us = (time.perf_counter_ns() - t0) / 1000
        times_us.append(elapsed_us)
        allocator.free(ids)

    print("| call_index | latency_us |")
    print("|---|---|")
    for i, t in enumerate(times_us):
        print(f"| {i} | {t:.2f} |")

    print()
    first_call = times_us[0]
    steady = times_us[5:]
    steady_median = statistics.median(steady)
    steady_p95 = sorted(steady)[int(0.95 * len(steady))]
    print(f"call[0]              : {first_call:.2f} us")
    print(f"median(call[5:])     : {steady_median:.2f} us")
    print(f"p95(call[5:])        : {steady_p95:.2f} us")
    print(f"ratio call[0] / median: {first_call / steady_median:.2f}x")
    print()
    if first_call / steady_median > 1.5:
        print(
            "VERDICT: JIT warmup is significant; call[0] >> steady-state median. "
            "On-disk Triton cache may have a partial miss for this kernel "
            "signature, or the per-process Triton runtime initialization is "
            "incurring meaningful cost."
        )
    else:
        print(
            "VERDICT: JIT warmup is NOT the dominant cost. call[0] is comparable "
            "to steady-state. The per-call cost is dispatch + driver enqueue, "
            "amortized across the sweep."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["1", "2", "3", "all"],
        required=True,
        help="which investigation phase to run",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=_REPO_ROOT / "research/avmp/v2/results",
        help="directory for torch.profiler Chrome traces (Phase 2)",
    )
    args = parser.parse_args(argv)

    if args.phase in ("1", "all"):
        phase1()
    if args.phase in ("2", "all"):
        phase2(args.profile_output)
    if args.phase in ("3", "all"):
        phase3()
    return 0


if __name__ == "__main__":
    sys.exit(main())
