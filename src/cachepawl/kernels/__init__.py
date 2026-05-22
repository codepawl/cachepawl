"""Triton kernel implementations for the AVMP allocator.

CUDA C++ extensions are deliberately deferred until the Triton path
proves insufficient. The kernels here back the
:class:`cachepawl.allocator.avmp.TritonAVMPAllocator` hardware
realization of the AVMP design (see
``research/avmp/v2/TRITON_ROADMAP.md``).

Public symbols are scaffolds: kernel bodies are ``pass`` and the
``launch_*`` wrappers raise :class:`NotImplementedError`. Bodies land
in Week 1 / Week 2 of the roadmap.
"""

from cachepawl.kernels.allocate import launch_zero_page, zero_page_kernel
from cachepawl.kernels.config import KernelLaunchConfig
from cachepawl.kernels.migrate import copy_region_kernel, launch_copy_region

__all__ = [
    "KernelLaunchConfig",
    "copy_region_kernel",
    "launch_copy_region",
    "launch_zero_page",
    "zero_page_kernel",
]
