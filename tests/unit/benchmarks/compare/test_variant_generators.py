"""Unit tests for the parameterized variant generators in `sweep.py`.

The v2 stage 1 batch_size sweep adds `generate_batch_size_variants` and
the `BATCHSIZE_SWEEP_VARIANTS` constant. These tests pin the label grammar
and the kwargs payload so downstream `_build_allocator` (which parses
kwargs) stays in sync with the generator.
"""

from __future__ import annotations

import pytest

from cachepawl.benchmarks.compare.sweep import (
    BATCHSIZE_SWEEP_VARIANTS,
    AllocatorVariant,
    generate_batch_size_variants,
)


def test_generate_batch_size_variants_default_powers_of_two() -> None:
    variants = generate_batch_size_variants()
    assert len(variants) == 9
    labels = [v.label for v in variants]
    assert labels == [
        "avmp_dynamic_b1",
        "avmp_dynamic_b2",
        "avmp_dynamic_b4",
        "avmp_dynamic_b8",
        "avmp_dynamic_b16",
        "avmp_dynamic_b32",
        "avmp_dynamic_b64",
        "avmp_dynamic_b128",
        "avmp_dynamic_b256",
    ]


def test_generate_batch_size_variants_kwargs_carry_mamba_and_batch() -> None:
    variants = generate_batch_size_variants(batch_sizes=(2, 4), mamba_ratio=0.5)
    assert len(variants) == 2
    for v in variants:
        assert v.allocator_name == "avmp_dynamic"
        assert dict(v.kwargs)["mamba_ratio"] == 0.5
        assert "migration_batch_size" in dict(v.kwargs)
    assert dict(variants[0].kwargs)["migration_batch_size"] == 2.0
    assert dict(variants[1].kwargs)["migration_batch_size"] == 4.0


def test_generate_batch_size_variants_custom_ratio() -> None:
    variants = generate_batch_size_variants(batch_sizes=(8,), mamba_ratio=0.7)
    assert dict(variants[0].kwargs)["mamba_ratio"] == 0.7


def test_batchsize_sweep_variants_has_twelve_unique_labels() -> None:
    """3 baselines + 9 batch variants, all distinct labels."""

    assert len(BATCHSIZE_SWEEP_VARIANTS) == 12
    labels = [v.label for v in BATCHSIZE_SWEEP_VARIANTS]
    assert len(set(labels)) == 12
    # Spot-check key entries.
    assert "padded_unified" in labels
    assert "fixed_dual_mr05" in labels
    assert "avmp_static_mr05" in labels
    assert "avmp_dynamic_b1" in labels
    assert "avmp_dynamic_b256" in labels


def test_batchsize_sweep_variants_have_compatible_allocator_names() -> None:
    """Every variant's allocator_name must be one of the four handled by
    `_build_allocator`. Mismatch surfaces as a validation error at sweep
    construction time, which makes for a noisy debug; pin here instead."""

    valid = {"padded_unified", "fixed_dual", "avmp_static", "avmp_dynamic"}
    for v in BATCHSIZE_SWEEP_VARIANTS:
        assert v.allocator_name in valid, (
            f"variant {v.label}: unknown allocator_name {v.allocator_name}"
        )


def test_generate_batch_size_variants_returns_immutable_tuple() -> None:
    """Output must be a tuple of AllocatorVariant instances so it can flow
    into the frozen-dataclass `SweepConfig.variants` field."""

    variants = generate_batch_size_variants()
    assert isinstance(variants, tuple)
    assert all(isinstance(v, AllocatorVariant) for v in variants)
    with pytest.raises(TypeError):
        variants[0] = AllocatorVariant(  # type: ignore[index]
            label="x", allocator_name="padded_unified", kwargs=()
        )
