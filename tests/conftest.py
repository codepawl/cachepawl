"""Shared pytest fixtures and gpu-marker auto-skip."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """True when at least one CUDA device is visible to torch."""

    return torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _skip_gpu_without_cuda(request: pytest.FixtureRequest) -> None:
    """Skip tests marked ``gpu`` when no CUDA device is visible."""

    if request.node.get_closest_marker("gpu") is None:
        return
    if not torch.cuda.is_available():
        pytest.skip("CUDA-capable GPU required")
