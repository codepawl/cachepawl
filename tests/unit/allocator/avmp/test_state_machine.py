"""Unit tests for :mod:`cachepawl.allocator.avmp.state`.

Covers the four documented compute_state outcomes from RFC 0002 section 4.1,
constructor validation, ring buffer eviction, transition counting, and the
regression that ``REBALANCING`` is never returned by ``compute_state`` in
v2 sub-PR 1.
"""

from __future__ import annotations

import pytest

from cachepawl.allocator.avmp.state import (
    VALID_POLLING_STRATEGIES,
    PoolPressureMonitor,
    PoolPressureState,
)


def test_enum_int_values_match_documented_encoding() -> None:
    """The integer values back ``current_pressure_state_code`` in stats."""

    assert PoolPressureState.BALANCED.value == 0
    assert PoolPressureState.KV_PRESSURED.value == 1
    assert PoolPressureState.SSM_PRESSURED.value == 2
    assert PoolPressureState.REBALANCING.value == 3


def test_valid_polling_strategies_set() -> None:
    assert frozenset({"on_allocate", "on_sample"}) == VALID_POLLING_STRATEGIES


def test_default_thresholds_match_rfc_0002() -> None:
    """RFC 0002 section 4.2: threshold_low=0.05, threshold_high=0.30."""

    monitor = PoolPressureMonitor()
    assert monitor.threshold_low == 0.05
    assert monitor.threshold_high == 0.30
    assert monitor.polling_strategy == "on_allocate"


@pytest.mark.parametrize(
    ("kv_free_ratio", "ssm_free_ratio", "expected"),
    [
        # Balanced: neither pool below threshold_low
        (0.5, 0.5, PoolPressureState.BALANCED),
        (1.0, 1.0, PoolPressureState.BALANCED),
        # KV pressured: kv below low AND ssm above high
        (0.01, 0.5, PoolPressureState.KV_PRESSURED),
        (0.04, 0.31, PoolPressureState.KV_PRESSURED),
        # SSM pressured: ssm below low AND kv above high
        (0.5, 0.01, PoolPressureState.SSM_PRESSURED),
        (0.31, 0.04, PoolPressureState.SSM_PRESSURED),
        # Both below threshold_low: ambiguous, falls through to BALANCED
        (0.01, 0.01, PoolPressureState.BALANCED),
        # One pool below low but other not above high: BALANCED (no rescue capacity)
        (0.01, 0.10, PoolPressureState.BALANCED),
        (0.10, 0.01, PoolPressureState.BALANCED),
        # Edge: exactly on threshold_low (strict <)
        (0.05, 0.5, PoolPressureState.BALANCED),
        # Edge: exactly on threshold_high (strict >)
        (0.01, 0.30, PoolPressureState.BALANCED),
    ],
)
def test_compute_state_returns_documented_outcomes(
    kv_free_ratio: float,
    ssm_free_ratio: float,
    expected: PoolPressureState,
) -> None:
    monitor = PoolPressureMonitor()
    assert monitor.compute_state(kv_free_ratio, ssm_free_ratio) is expected


def test_compute_state_never_returns_rebalancing_in_sub_pr_1() -> None:
    """REBALANCING is reserved for the pool to set when migration starts (sub-PR 2)."""

    monitor = PoolPressureMonitor()
    grid = [i / 20.0 for i in range(21)]  # 0.00, 0.05, ..., 1.00
    for kv in grid:
        for ssm in grid:
            assert monitor.compute_state(kv, ssm) is not PoolPressureState.REBALANCING


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (0.0, 0.30),  # threshold_low must be strictly positive
        (0.05, 1.0),  # threshold_high must be strictly less than 1
        (0.30, 0.05),  # low must be strictly less than high
        (0.30, 0.30),  # equal is rejected
        (-0.01, 0.5),  # negative low
        (0.5, 1.5),  # high above 1
    ],
)
def test_constructor_rejects_invalid_thresholds(low: float, high: float) -> None:
    with pytest.raises(ValueError, match="thresholds must satisfy"):
        PoolPressureMonitor(threshold_low=low, threshold_high=high)


def test_constructor_rejects_unknown_polling_strategy() -> None:
    with pytest.raises(ValueError, match="polling_strategy"):
        PoolPressureMonitor(polling_strategy="never")


def test_constructor_rejects_nonpositive_ring_buffer_size() -> None:
    with pytest.raises(ValueError, match="ring_buffer_size must be positive"):
        PoolPressureMonitor(ring_buffer_size=0)


def test_ring_buffer_drops_oldest_when_full() -> None:
    monitor = PoolPressureMonitor(ring_buffer_size=3)
    for i in range(5):
        monitor.record_transition(
            PoolPressureState.BALANCED,
            PoolPressureState.KV_PRESSURED,
            timestamp_ns=i,
        )
    timestamps = [t for t, _, _ in monitor.transitions]
    # Oldest two (0, 1) evicted; (2, 3, 4) remain in insertion order.
    assert timestamps == [2, 3, 4]


def test_transition_count_over_known_sequence() -> None:
    monitor = PoolPressureMonitor()
    transitions = [
        (PoolPressureState.BALANCED, PoolPressureState.KV_PRESSURED),
        (PoolPressureState.KV_PRESSURED, PoolPressureState.BALANCED),
        (PoolPressureState.BALANCED, PoolPressureState.KV_PRESSURED),
        (PoolPressureState.KV_PRESSURED, PoolPressureState.BALANCED),
        (PoolPressureState.BALANCED, PoolPressureState.SSM_PRESSURED),
    ]
    for ts, (prev, new) in enumerate(transitions):
        monitor.record_transition(prev, new, timestamp_ns=ts)

    assert monitor.transition_count(
        PoolPressureState.BALANCED, PoolPressureState.KV_PRESSURED
    ) == 2
    assert monitor.transition_count(
        PoolPressureState.KV_PRESSURED, PoolPressureState.BALANCED
    ) == 2
    assert monitor.transition_count(
        PoolPressureState.BALANCED, PoolPressureState.SSM_PRESSURED
    ) == 1
    assert monitor.transition_count(
        PoolPressureState.SSM_PRESSURED, PoolPressureState.BALANCED
    ) == 0
