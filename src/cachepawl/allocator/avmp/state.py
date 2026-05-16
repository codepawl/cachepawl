"""State machine for AVMP v2 dynamic pool rebalancing.

This module is observation-only in v2 sub-PR 1. :meth:`PoolPressureMonitor.compute_state`
returns a :class:`PoolPressureState` based on the configured free-fraction
watermarks, and the monitor keeps a bounded ring buffer of state transitions
for downstream observability. The migration path that takes a ``KV_PRESSURED``
or ``SSM_PRESSURED`` reading and transitions the pool to ``REBALANCING`` lands
in v2 sub-PR 2 (RFC 0002 section 7).

The four-state enum is the design from RFC 0002 section 4.1. ``REBALANCING``
is reserved by :meth:`PoolPressureMonitor.compute_state` and never returned
by it; the pool drives the transition into ``REBALANCING`` itself when
migration starts.
"""

from __future__ import annotations

import collections
import enum


class PoolPressureState(enum.IntEnum):
    """Pool pressure states from RFC 0002 section 4.1.

    ``IntEnum`` so the value flows directly to ``float()`` when surfaced via
    ``get_allocator_stats``'s ``current_pressure_state_code`` key.
    """

    BALANCED = 0
    KV_PRESSURED = 1
    SSM_PRESSURED = 2
    REBALANCING = 3


VALID_POLLING_STRATEGIES: frozenset[str] = frozenset({"on_allocate", "on_sample"})


class PoolPressureMonitor:
    """Computes pressure state and records transitions in a bounded ring buffer.

    The monitor is passive: the caller (typically :class:`AsymmetricVirtualPool`)
    invokes :meth:`compute_state` whenever it wants a fresh reading and calls
    :meth:`record_transition` to log a transition into the ring buffer. The
    ``polling_strategy`` argument is stored for v2 sub-PR 3 dispatch and is
    not acted on in this PR; both accepted values keep the API stable so
    downstream code can already construct monitors with either strategy.
    """

    def __init__(
        self,
        threshold_low: float = 0.05,
        threshold_high: float = 0.30,
        polling_strategy: str = "on_allocate",
        ring_buffer_size: int = 128,
    ) -> None:
        if not 0.0 < threshold_low < threshold_high < 1.0:
            raise ValueError(
                "PoolPressureMonitor: thresholds must satisfy "
                "0.0 < threshold_low < threshold_high < 1.0, got "
                f"threshold_low={threshold_low}, threshold_high={threshold_high}"
            )
        if polling_strategy not in VALID_POLLING_STRATEGIES:
            raise ValueError(
                f"PoolPressureMonitor: polling_strategy={polling_strategy!r} is not one of "
                f"{sorted(VALID_POLLING_STRATEGIES)}"
            )
        if ring_buffer_size <= 0:
            raise ValueError(
                f"PoolPressureMonitor: ring_buffer_size must be positive, got {ring_buffer_size}"
            )
        self._threshold_low = threshold_low
        self._threshold_high = threshold_high
        self._polling_strategy = polling_strategy
        self._transitions: collections.deque[tuple[int, PoolPressureState, PoolPressureState]] = (
            collections.deque(maxlen=ring_buffer_size)
        )

    @property
    def threshold_low(self) -> float:
        return self._threshold_low

    @property
    def threshold_high(self) -> float:
        return self._threshold_high

    @property
    def polling_strategy(self) -> str:
        return self._polling_strategy

    def compute_state(self, kv_free_ratio: float, ssm_free_ratio: float) -> PoolPressureState:
        """Map the free-fraction pair to a pressure state.

        Returns ``KV_PRESSURED`` iff KV is below ``threshold_low`` AND SSM is
        above ``threshold_high``; ``SSM_PRESSURED`` for the mirror; otherwise
        ``BALANCED``. ``REBALANCING`` is never returned in v2 sub-PR 1; the
        pool drives that transition itself when migration starts (sub-PR 2).
        """

        if kv_free_ratio < self._threshold_low and ssm_free_ratio > self._threshold_high:
            return PoolPressureState.KV_PRESSURED
        if ssm_free_ratio < self._threshold_low and kv_free_ratio > self._threshold_high:
            return PoolPressureState.SSM_PRESSURED
        return PoolPressureState.BALANCED

    def record_transition(
        self,
        prev: PoolPressureState,
        new: PoolPressureState,
        timestamp_ns: int,
    ) -> None:
        """Append a transition to the bounded ring buffer."""

        self._transitions.append((timestamp_ns, prev, new))

    def transition_count(
        self,
        from_state: PoolPressureState,
        to_state: PoolPressureState,
    ) -> int:
        """Count transitions ``from_state -> to_state`` currently in the buffer."""

        return sum(1 for _, p, n in self._transitions if p is from_state and n is to_state)

    @property
    def transitions(self) -> tuple[tuple[int, PoolPressureState, PoolPressureState], ...]:
        """Read-only snapshot of buffered transitions, oldest first."""

        return tuple(self._transitions)
