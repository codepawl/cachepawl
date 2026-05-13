"""Cachepawl: hybrid KV and SSM cache allocator for Mamba-Transformer-MoE models."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("cachepawl")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
