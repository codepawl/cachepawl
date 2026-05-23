"""Runtime and GPU metadata capture for planner benchmarks."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

import numpy as np
import torch

import cachepawl
from cachepawl.bench.result_schema import GpuMetadata, Metadata


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    """Runtime metadata captured without importing optional vLLM."""

    gpu: GpuMetadata
    metadata: Metadata


def capture_gpu_metadata(
    *,
    total_memory_bytes_override: int | None = None,
    name_override: str | None = None,
) -> GpuMetadata:
    """Capture GPU metadata with CPU-safe fallbacks."""

    if total_memory_bytes_override is not None and total_memory_bytes_override < 0:
        raise ValueError("total_memory_bytes_override must be non-negative when provided")

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    if not cuda_available:
        return GpuMetadata(
            name=name_override,
            total_memory_bytes=total_memory_bytes_override,
            compute_capability=None,
            cuda_available=False,
            device_count=0,
        )

    props = torch.cuda.get_device_properties(0)
    capability = (int(props.major), int(props.minor))
    return GpuMetadata(
        name=name_override if name_override is not None else str(props.name),
        total_memory_bytes=(
            total_memory_bytes_override
            if total_memory_bytes_override is not None
            else int(props.total_memory)
        ),
        compute_capability=capability,
        cuda_available=True,
        device_count=device_count,
    )


def capture_environment(
    *,
    total_memory_bytes_override: int | None = None,
    name_override: str | None = None,
) -> RuntimeEnvironment:
    """Capture benchmark runtime metadata without requiring CUDA or vLLM."""

    gpu = capture_gpu_metadata(
        total_memory_bytes_override=total_memory_bytes_override,
        name_override=name_override,
    )
    return RuntimeEnvironment(
        gpu=gpu,
        metadata={
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cachepawl_version": cachepawl.__version__,
            "cuda_version": torch.version.cuda,
            "vllm_version": _optional_package_version("vllm"),
        },
    )


def _optional_package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None
