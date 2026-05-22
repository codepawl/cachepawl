"""Parity harness placeholder: TritonAVMPAllocator vs Python AVMP.

The parity smoke (9 cells = 3 workloads x 1 spec x 1 byte-size x 3 seeds)
runs in Week 2 per ``research/avmp/v2/TRITON_ROADMAP.md`` section 3. Pass
criteria per cell: ``total_oom`` identical within +/- 1 event,
``effective_batch_size_p50`` and ``fragmentation_p50`` within 1% of the
Python baseline.

This file scaffolds the harness so test collection sees it. The actual
parity assertions land alongside the Week 1 / Week 2 kernel bodies.
"""

from __future__ import annotations

import pytest

from cachepawl.allocator.avmp import AsymmetricVirtualPool, TritonAVMPAllocator


def test_triton_allocator_is_subclass_of_python_baseline() -> None:
    """Inheritance choice (TRITON_ROADMAP.md section 1) is enforced.

    If a future refactor splits these into siblings, the parity harness
    needs a different fixture wiring than this scaffold assumes; failing
    this assertion is the signal to revisit the decision in the
    roadmap.
    """

    assert issubclass(TritonAVMPAllocator, AsymmetricVirtualPool)


@pytest.mark.gpu
@pytest.mark.skip(reason="parity smoke runs in Week 2; see TRITON_ROADMAP.md section 3")
def test_parity_smoke_9_cells() -> None:
    """Smoke parity placeholder.

    Week 2 will:

    1. Build the 9-cell grid (3 workloads x 3 seeds at 1 spec, 1
       byte-size) using the existing benchmark runner from
       :mod:`cachepawl.benchmarks`.
    2. Run each cell twice: once with the registered ``avmp_dynamic``
       Python allocator and once with :class:`TritonAVMPAllocator`.
    3. Assert ``total_oom``, ``effective_batch_size_p50``, and
       ``fragmentation_p50`` agree per the tolerances above.
    """
