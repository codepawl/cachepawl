"""Device and VRAM introspection helpers.

torch is imported lazily inside each helper to keep ``import cachepawl``
cheap and to let CPU-only environments load the package without
triggering CUDA initialization.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VramInfo:
    """Snapshot of VRAM occupancy on a single CUDA device."""

    total_bytes: int
    free_bytes: int


def get_device() -> str:
    """Return ``"cuda"`` when a CUDA device is visible, else ``"cpu"``."""

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def vram_info(device_index: int = 0) -> VramInfo:
    """Return free and total VRAM in bytes for the given CUDA device.

    Raises ``NotImplementedError`` until the device probe is wired up.
    """

    raise NotImplementedError(
        f"vram_info: implement once the device probe lands (device_index={device_index})."
    )


def cuda_capability(device_index: int = 0) -> tuple[int, int]:
    """Return the (major, minor) CUDA compute capability for the device.

    Raises ``NotImplementedError`` until the device probe is wired up.
    """

    raise NotImplementedError(
        f"cuda_capability: implement once the device probe lands (device_index={device_index})."
    )
