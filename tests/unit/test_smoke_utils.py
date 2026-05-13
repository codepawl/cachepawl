"""Smoke test for the utils submodule public surface."""

from __future__ import annotations

import pytest

from cachepawl.utils import VramInfo, cuda_capability, get_device, vram_info


def test_get_device_returns_cpu_or_cuda() -> None:
    assert get_device() in {"cpu", "cuda"}


def test_vram_info_dataclass_holds_byte_counts() -> None:
    info = VramInfo(total_bytes=12 * 1024**3, free_bytes=10 * 1024**3)
    assert info.total_bytes == 12 * 1024**3
    assert info.free_bytes == 10 * 1024**3


def test_vram_info_helper_is_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="vram_info"):
        vram_info(0)


def test_cuda_capability_helper_is_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="cuda_capability"):
        cuda_capability(0)
