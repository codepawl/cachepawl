"""Smoke test for the quant submodule public surface."""

from __future__ import annotations

import pytest

from cachepawl.quant import DType, bytes_per_element


@pytest.mark.parametrize(
    ("dtype", "width"),
    [
        (DType.FP16, 2.0),
        (DType.BF16, 2.0),
        (DType.INT8, 1.0),
        (DType.FP8_E4M3, 1.0),
        (DType.FP8_E5M2, 1.0),
        (DType.FP4, 0.5),
    ],
)
def test_bytes_per_element_widths(dtype: DType, width: float) -> None:
    assert bytes_per_element(dtype) == width


def test_all_dtype_members_have_a_width() -> None:
    for dtype in DType:
        assert bytes_per_element(dtype) > 0
