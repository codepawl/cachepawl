"""Tests for planner-only cache memory estimates."""

from __future__ import annotations

import pytest

from cachepawl.bench.planner_baselines import (
    estimate_avmp_static,
    estimate_planner,
    estimate_vllm_style_padded,
    overestimation_ratio,
    wasted_fraction,
)
from cachepawl.bench.synthetic_workloads import generate_synthetic_workload


def test_vllm_style_padded_reserves_at_least_useful_bytes() -> None:
    workload = generate_synthetic_workload("mixed", seed=1, num_requests=8)
    estimate = estimate_vllm_style_padded(workload)
    assert estimate.backend == "vllm-style-padded"
    assert estimate.reserved_bytes >= estimate.useful_bytes
    assert estimate.estimated_bytes == estimate.reserved_bytes
    assert estimate.overestimation_ratio >= 1.0
    assert 0.0 <= estimate.wasted_fraction <= 1.0
    assert estimate.metadata["planner_model"] == "uniform-page-padding"


def test_avmp_has_lower_waste_than_padded_baseline_on_hybrid_workload() -> None:
    workload = generate_synthetic_workload("mixed", seed=1, num_requests=8)
    padded = estimate_vllm_style_padded(workload)
    avmp = estimate_avmp_static(workload)
    assert avmp.backend == "cachepawl-avmp"
    assert avmp.reserved_bytes <= padded.reserved_bytes
    assert avmp.overestimation_ratio <= padded.overestimation_ratio
    assert avmp.wasted_fraction <= padded.wasted_fraction
    assert avmp.overestimation_ratio == pytest.approx(1.0, rel=0.01)
    assert avmp.metadata["planner_model"] == "native-kv-page-ssm-block"


def test_metric_helpers_distinguish_ratio_from_fraction() -> None:
    estimated_bytes = 614_465_536
    useful_bytes = 196_526_080
    assert overestimation_ratio(
        estimated_bytes=estimated_bytes,
        useful_bytes=useful_bytes,
    ) == pytest.approx(3.1266, rel=1e-4)
    assert wasted_fraction(
        estimated_bytes=estimated_bytes,
        useful_bytes=useful_bytes,
    ) == pytest.approx(0.6802, rel=1e-4)


def test_planner_estimates_are_deterministic_for_fixed_input() -> None:
    workload = generate_synthetic_workload("long-heavy", seed=42, num_requests=6)
    first = estimate_planner("vllm-style-padded", workload)
    second = estimate_planner("vllm-style-padded", workload)
    assert first == second


def test_virtual_oom_uses_reserved_bytes_threshold() -> None:
    workload = generate_synthetic_workload("short-heavy", seed=1, num_requests=4)
    no_oom = estimate_avmp_static(workload, gpu_total_bytes=10**12)
    oom = estimate_avmp_static(workload, gpu_total_bytes=1)
    assert no_oom.virtual_oom is False
    assert oom.virtual_oom is True
