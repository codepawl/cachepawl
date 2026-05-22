"""Launch configuration dataclass for the AVMP Triton kernels.

Reference: ``research/avmp/v2/TRITON_ROADMAP.md`` section 2 (kernel API
design and RTX 3060 hardware envelope).

Pure dataclass. No Triton import, so this module is safe to import on
non-Linux platforms where Triton is unavailable. The kernel modules
under :mod:`cachepawl.kernels` import Triton directly and are
import-gated by ``platform_system == 'Linux'`` via the dependency
declaration in ``pyproject.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["KernelLaunchConfig"]


@dataclass(frozen=True, slots=True)
class KernelLaunchConfig:
    """Compile-time and launch-time parameters for one Triton kernel call.

    Field semantics:

    - ``grid``: 1D, 2D, or 3D grid passed to ``kernel[grid](...)``.
      Computed by the caller from the kernel-specific work size (e.g.
      ``(triton.cdiv(size_bytes, block_size),)`` for ``zero_page_kernel``).
    - ``block_size``: bytes per program for the AVMP kernels; passed as
      the ``BLOCK_SIZE`` constexpr to the kernel. Default tuning target
      is 4096 on RTX 3060 (CC 8.6) per the roadmap; smaller values trade
      coalescing for finer tail granularity.
    - ``num_warps``: passed to the kernel launch as
      ``kernel[grid](..., num_warps=num_warps)``. Default of 4 matches
      Triton's default for streaming-store kernels.
    - ``num_stages``: pipeline depth for the kernel launch. Default of 2
      is sufficient for pure store kernels; copy kernels may benefit
      from 3+ once tuned.
    """

    grid: tuple[int, ...]
    block_size: int
    num_warps: int = 4
    num_stages: int = 2
