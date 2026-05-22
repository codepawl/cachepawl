"""Parity harness: TritonAVMPAllocator vs Python avmp_dynamic baseline.

The 9-cell parity smoke (3 workloads x 1 spec x 1 total_bytes x 3 seeds)
runs in Week 2 per ``research/avmp/v2/TRITON_ROADMAP.md`` section 3.
Pass criteria per cell:

- ``oom_count`` identical (event-deterministic field; the simulator
  does not touch GPU randomness, only the allocator does, and the
  allocator's OOM emission is a function of capacity and the event
  stream, both deterministic for a fixed seed).
- ``effective_batch_size_p50`` within 1% relative difference.

The reference baseline is ``AsymmetricVirtualPool`` with the same
constructor kwargs as the registered ``avmp_dynamic`` factory
(``rebalance_enabled=True`` plus defaults), overriding only
``total_bytes=4 GiB`` from the registry's 8 GiB default so the 2x
physical footprint (RFC 0002 section 4.3) fits in a 12 GiB RTX 3060.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool, TritonAVMPAllocator
from cachepawl.benchmarks import (
    PRESETS,
    HybridModelSpec,
    LayerKind,
    LayerSpec,
    WorkloadSpec,
    run_benchmark,
)

if TYPE_CHECKING:
    from cachepawl.allocator.base import Allocator


_PARITY_TOTAL_BYTES = 4 * 1024**3  # 4 GiB; safe for RTX 3060's 12 GiB with 2x footprint.
_PARITY_WORKLOADS = ("uniform_short", "mixed_long", "agentic_burst")
_PARITY_SEEDS = (20260520, 20260521, 20260522)
_REL_TOL = 0.01  # 1% per TRITON_ROADMAP.md section 3.


def test_triton_allocator_is_subclass_of_python_baseline() -> None:
    """Inheritance choice (TRITON_ROADMAP.md section 1) is enforced.

    If a future refactor splits these into siblings, the parity harness
    needs different fixture wiring than this test assumes; failing
    this assertion is the signal to revisit the decision in the
    roadmap.
    """

    assert issubclass(TritonAVMPAllocator, AsymmetricVirtualPool)


def _hybrid_spec_from_workload(spec: WorkloadSpec) -> HybridModelSpec:
    """Mirror ``cachepawl.benchmarks._hybrid_spec_from_workload``.

    Recreated here rather than imported because the registry helper is
    private. Layer interleaving uses the same ``attn_every`` formula so
    the model layout matches the registered factories byte-for-byte.
    """

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


def _make_python_baseline(workload: WorkloadSpec, device: torch.device) -> AsymmetricVirtualPool:
    return AsymmetricVirtualPool(
        model_spec=_hybrid_spec_from_workload(workload),
        total_bytes=_PARITY_TOTAL_BYTES,
        device=device,
        rebalance_enabled=True,
    )


def _make_triton_allocator(workload: WorkloadSpec, device: torch.device) -> TritonAVMPAllocator:
    return TritonAVMPAllocator(
        model_spec=_hybrid_spec_from_workload(workload),
        total_bytes=_PARITY_TOTAL_BYTES,
        device=device,
        rebalance_enabled=True,
    )


def _run_cell(
    allocator: Allocator,
    workload: WorkloadSpec,
    output_dir: Path,
    name: str,
) -> tuple[int, float]:
    run = run_benchmark(
        allocator,
        workload,
        allocator_name=name,
        output_dir=output_dir,
        device="cuda",
    )
    return run.metrics.oom_count, run.metrics.effective_batch_size_p50


@pytest.mark.gpu
@pytest.mark.slow
def test_parity_smoke_9_cells(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """9-cell smoke: TritonAVMPAllocator matches Python avmp_dynamic.

    Cells: 3 workloads x 3 seeds. Per cell, run both allocators through
    the benchmark harness and compare OOM count exactly and
    effective_batch_size_p50 within 1%.

    The 18 invocations of ``run_benchmark`` (9 cells x 2 allocators)
    drive the harness on CUDA; the simulator is event-deterministic so
    runs at the same seed produce the same event stream, and the
    OOM-count match is exact (not +/- 1).

    Prints a Markdown comparison table to stdout for paper-section
    reuse. Capture-suppressed output is visible with ``pytest -s``.
    """

    device = torch.device("cuda")
    rows: list[tuple[str, int, int, int, float, float, float]] = []
    for workload_name in _PARITY_WORKLOADS:
        base_workload = PRESETS[workload_name]
        for seed in _PARITY_SEEDS:
            workload = dataclasses.replace(base_workload, seed=seed)
            baseline_alloc = _make_python_baseline(workload, device)
            baseline_oom, baseline_p50 = _run_cell(
                baseline_alloc, workload, tmp_path / "baseline", "avmp_dynamic_baseline"
            )
            triton_alloc = _make_triton_allocator(workload, device)
            triton_oom, triton_p50 = _run_cell(
                triton_alloc, workload, tmp_path / "triton", "avmp_dynamic_triton"
            )
            rel_diff = abs(triton_p50 - baseline_p50) / baseline_p50 if baseline_p50 > 0 else 0.0
            rows.append(
                (workload_name, seed, baseline_oom, triton_oom, baseline_p50, triton_p50, rel_diff)
            )
            assert triton_oom == baseline_oom, (
                f"OOM count mismatch on {workload_name} seed={seed}: "
                f"baseline={baseline_oom}, triton={triton_oom}"
            )
            assert rel_diff <= _REL_TOL, (
                f"effective_batch_size_p50 drift on {workload_name} seed={seed}: "
                f"baseline={baseline_p50:.4f}, triton={triton_p50:.4f}, "
                f"rel_diff={rel_diff:.4%} > {_REL_TOL:.0%}"
            )

    with capsys.disabled():
        print("\n### 9-cell parity smoke: TritonAVMPAllocator vs avmp_dynamic (baseline)\n")
        header = (
            "| Workload | Seed | Baseline OOM | Triton OOM | Baseline p50 | Triton p50 | rel diff |"
        )
        print(header)
        print("|---|---|---|---|---|---|---|")
        for workload_name, seed, base_oom, triton_oom, base_p50, triton_p50, rel in rows:
            print(
                f"| {workload_name} | {seed} | {base_oom} | {triton_oom} | "
                f"{base_p50:.4f} | {triton_p50:.4f} | {rel:.4%} |"
            )
