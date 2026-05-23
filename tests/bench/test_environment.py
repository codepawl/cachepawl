"""Tests for CPU-safe environment capture."""

from __future__ import annotations

import sys

from cachepawl.bench.environment import capture_environment, capture_gpu_metadata


def test_capture_gpu_metadata_supports_cpu_fallback_override() -> None:
    gpu = capture_gpu_metadata(
        total_memory_bytes_override=12 * 1024**3,
        name_override="NVIDIA GeForce RTX 3060",
    )
    assert gpu.name == "NVIDIA GeForce RTX 3060"
    assert gpu.total_memory_bytes == 12 * 1024**3
    assert gpu.device_count >= 0
    if not gpu.cuda_available:
        assert gpu.compute_capability is None


def test_capture_environment_does_not_require_vllm_import() -> None:
    before = "vllm" in sys.modules
    env = capture_environment()
    assert "python_version" in env.metadata
    assert "torch_version" in env.metadata
    assert "vllm_version" in env.metadata
    assert ("vllm" in sys.modules) is before
