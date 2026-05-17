"""Unit tests for the stage 2 threshold variant generator in `sweep.py`.

Pins the label slug grammar, the kwargs payload, and the
``THRESHOLD_SWEEP_STAGE2_VARIANTS`` constant so downstream
``_build_allocator`` (which parses these kwargs) stays in sync.
"""

from __future__ import annotations

import pytest
import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.benchmarks.compare.sweep import (
    THRESHOLD_SWEEP_STAGE2_VARIANTS,
    AllocatorVariant,
    _build_allocator,
    _threshold_slug,
    generate_threshold_variants,
)
from cachepawl.models.spec import JAMBA_1_5_MINI_REF


def test_threshold_slug_round_trip() -> None:
    assert _threshold_slug(0.02) == "002"
    assert _threshold_slug(0.05) == "005"
    assert _threshold_slug(0.10) == "010"
    assert _threshold_slug(0.20) == "020"
    assert _threshold_slug(0.30) == "030"


def test_generate_threshold_variants_defaults_return_four_variants() -> None:
    variants = generate_threshold_variants()
    assert len(variants) == 4
    labels = [v.label for v in variants]
    assert labels == [
        "avmp_dynamic_b128_th_high_010",
        "avmp_dynamic_b128_th_high_020",
        "avmp_dynamic_b128_th_low_002",
        "avmp_dynamic_b128_th_low_010",
    ]


def test_generate_threshold_variants_kwargs_carry_all_four_knobs() -> None:
    """Each variant carries mamba_ratio, migration_batch_size, threshold_low,
    threshold_high. The Hypothesis A half holds threshold_low at 0.05; the
    Hypothesis B half holds threshold_high at 0.30."""

    variants = generate_threshold_variants()
    for v in variants:
        kw = dict(v.kwargs)
        assert kw["mamba_ratio"] == 0.5
        assert kw["migration_batch_size"] == 128.0
        assert "threshold_low" in kw
        assert "threshold_high" in kw

    by_label = {v.label: dict(v.kwargs) for v in variants}
    # Hypothesis A: vary threshold_high.
    assert by_label["avmp_dynamic_b128_th_high_010"]["threshold_high"] == 0.10
    assert by_label["avmp_dynamic_b128_th_high_010"]["threshold_low"] == 0.05
    assert by_label["avmp_dynamic_b128_th_high_020"]["threshold_high"] == 0.20
    assert by_label["avmp_dynamic_b128_th_high_020"]["threshold_low"] == 0.05
    # Hypothesis B: vary threshold_low.
    assert by_label["avmp_dynamic_b128_th_low_002"]["threshold_low"] == 0.02
    assert by_label["avmp_dynamic_b128_th_low_002"]["threshold_high"] == 0.30
    assert by_label["avmp_dynamic_b128_th_low_010"]["threshold_low"] == 0.10
    assert by_label["avmp_dynamic_b128_th_low_010"]["threshold_high"] == 0.30


def test_generate_threshold_variants_custom_batch_size() -> None:
    """Changing batch_size changes the label slug and the kwargs value."""

    variants = generate_threshold_variants(batch_size=64, th_high_grid=(0.15,), th_low_grid=())
    assert len(variants) == 1
    assert variants[0].label == "avmp_dynamic_b64_th_high_015"
    assert dict(variants[0].kwargs)["migration_batch_size"] == 64.0


def test_threshold_sweep_stage2_variants_has_seven_unique_labels() -> None:
    """3 baselines + 4 threshold variants, all distinct."""

    assert len(THRESHOLD_SWEEP_STAGE2_VARIANTS) == 7
    labels = [v.label for v in THRESHOLD_SWEEP_STAGE2_VARIANTS]
    assert len(set(labels)) == 7
    assert "padded_unified" in labels
    assert "fixed_dual_mr05" in labels
    assert "avmp_static_mr05" in labels
    assert "avmp_dynamic_b128_th_high_010" in labels
    assert "avmp_dynamic_b128_th_low_010" in labels


def test_threshold_sweep_stage2_variants_have_compatible_allocator_names() -> None:
    valid = {"padded_unified", "fixed_dual", "avmp_static", "avmp_dynamic"}
    for v in THRESHOLD_SWEEP_STAGE2_VARIANTS:
        assert v.allocator_name in valid, (
            f"variant {v.label}: unknown allocator_name {v.allocator_name}"
        )


def test_generate_threshold_variants_returns_immutable_tuple() -> None:
    variants = generate_threshold_variants()
    assert isinstance(variants, tuple)
    assert all(isinstance(v, AllocatorVariant) for v in variants)
    with pytest.raises(TypeError):
        variants[0] = AllocatorVariant(  # type: ignore[index]
            label="x", allocator_name="padded_unified", kwargs=()
        )


def test_build_allocator_accepts_threshold_kwargs() -> None:
    """Regression test: _build_allocator pops threshold_low and threshold_high
    from kwargs without raising. Tests the kwargs-pop extension that PR #16's
    generate_batch_size_variants did not need."""

    for variant in generate_threshold_variants():
        pool = _build_allocator(
            variant,
            JAMBA_1_5_MINI_REF,
            total_bytes=1024 * 1024 * 1024,
            device=torch.device("cpu"),
        )
        assert isinstance(pool, AsymmetricVirtualPool)
        # Reach into instance state to confirm the thresholds applied.
        kw = dict(variant.kwargs)
        assert pool._threshold_low == kw["threshold_low"]
        assert pool._threshold_high == kw["threshold_high"]
