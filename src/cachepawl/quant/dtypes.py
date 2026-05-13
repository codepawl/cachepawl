"""Cache element dtypes spanning float, integer, and sub-byte formats."""

from __future__ import annotations

import enum


class DType(enum.Enum):
    """Supported cache element dtypes.

    FP4 is a placeholder for future packed sub-byte storage. The byte
    width helper below reports 0.5 for it.
    """

    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP4 = "fp4"


def bytes_per_element(dtype: DType) -> float:
    """Return the storage width of ``dtype`` in bytes.

    Returns 0.5 for FP4 to reflect packed two-elements-per-byte storage.
    """

    match dtype:
        case DType.FP16 | DType.BF16:
            return 2.0
        case DType.INT8 | DType.FP8_E4M3 | DType.FP8_E5M2:
            return 1.0
        case DType.FP4:
            return 0.5
