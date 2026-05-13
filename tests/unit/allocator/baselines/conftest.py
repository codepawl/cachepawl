"""Shared fixtures for the baseline allocator tests."""

from __future__ import annotations

import pytest
import torch

from cachepawl.models.spec import JAMBA_1_5_MINI_REF, MAMBA2_1B3_REF, HybridModelSpec


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def jamba_spec() -> HybridModelSpec:
    return JAMBA_1_5_MINI_REF


@pytest.fixture
def synthetic_large_state_spec() -> HybridModelSpec:
    """Synthetic Mamba-2-1.3B-like spec. SSM state size is large enough
    that PaddedUnifiedPool wastes a lot of memory on attention pages."""

    return MAMBA2_1B3_REF
