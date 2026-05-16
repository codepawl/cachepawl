"""Stateful property-based tests for :class:`PaddedUnifiedPool`.

Drives a single unified page table through allocate / free / partial-
free rules and asserts the documented baseline invariants:

- Pool capacity conservation: ``num_pages_used + num_pages_free ==
  num_pages_total`` at every step.
- ``padding_waste_bytes`` stays non-negative.
- The LRU tracker carries no entry with an empty ``page_ids`` list
  (the 79a98b0 lesson, the same regression net the baseline tests
  pin).
- After free (no internal eviction), ``padding_waste_bytes`` cannot
  grow. This is a transition property; tracked per-rule.

The pool routes alloc/free identically regardless of layer kind, but
its waste accounting subtracts the per-kind logical size from the
unified page size, so both kinds are exercised.

Hypothesis finding during PR hardening
=======================================

The hardening PR's initial test claimed ``padding_waste_bytes`` is
monotonically non-decreasing during allocate. Hypothesis falsified that
in 5.7 s with the shrunk sequence below on the Jamba-1.5-Mini spec at
4 MiB total budget. The trace was hand-verified to match the
documented implementation formula; this is NOT an allocator bug.

Sizes on the test spec:
``page_size_bytes = align_up(max(attn_logical=65536, ssm_logical=262144)) = 262144``,
so total = 16 pages, attn waste per page = 262144 - 65536 = 196608,
ssm waste per page = 0. Eviction policy is LRU.

Falsifying sequence and per-step deltas:

::

    step 0: empty                                  used=0  waste=0
    step 1: allocate(5, ATTENTION)                 used=5  waste=983040
            (+5 * 196608 attn-waste)
    step 2: allocate(7, ATTENTION)                 used=12 waste=2359296
            (+7 * 196608 attn-waste)
    step 3: allocate(5, MAMBA2)                    used=12 waste=1376256
            CapacityError -> evict request 1 (LRU oldest):
              5 attn pages freed -> waste -= 5 * 196608 = 983040
              used 12 -> 7
            retry alloc(5) succeeds:
              5 ssm pages allocated -> waste += 5 * 0 = 0
              used 7 -> 12
            net delta during step 3: -983040 (= -2359296 + 1376256)

The net-decrease happens because the evicted request had high per-page
waste (attention pages on a unified page sized for SSM) while the new
allocation has zero per-page waste (SSM pages match the page size). The
allocator's accounting is correct in both directions; the original
test invariant simply did not account for allocate-with-internal-
eviction. The reproducer script lives only in commit history; the
above trace is the durable record.
"""

from __future__ import annotations

import hypothesis.strategies as st
import torch
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from cachepawl.allocator.baselines import PaddedUnifiedPool
from cachepawl.models.spec import JAMBA_1_5_MINI_REF, LayerKind

_TOTAL_4_MIB = 4 * 1024 * 1024


class PaddedUnifiedStateMachine(RuleBasedStateMachine):
    """Random workload over PaddedUnifiedPool with invariant + transition checks."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = PaddedUnifiedPool(
            model_spec=JAMBA_1_5_MINI_REF,
            total_bytes=_TOTAL_4_MIB,
            device=torch.device("cpu"),
        )
        self.live: dict[int, list[int]] = {}
        self.next_request_id = 1

    @rule(
        count=st.integers(min_value=1, max_value=8),
        kind=st.sampled_from([LayerKind.ATTENTION, LayerKind.MAMBA2]),
    )
    def allocate_pages(self, count: int, kind: LayerKind) -> None:
        self.pool.set_current_layer_kind(kind)
        self.pool.set_current_request_id(self.next_request_id)
        try:
            ids = self.pool.allocate(count, dtype_bytes=2)
        except torch.cuda.OutOfMemoryError:
            self.next_request_id += 1
            return
        if ids:
            self.live[self.next_request_id] = ids
        self.next_request_id += 1

    @precondition(lambda self: bool(self.live))
    @rule(data=st.data())
    def free_request(self, data: st.DataObject) -> None:
        rid = data.draw(st.sampled_from(list(self.live)))
        ids = self.live.pop(rid)
        prev_waste = self.pool._padding_waste_bytes
        self.pool.free(ids)
        # Transition invariant: waste only drops on free (or stays equal
        # when freeing pages whose logical_bytes == page_size_bytes).
        assert self.pool._padding_waste_bytes <= prev_waste

    @precondition(lambda self: bool(self.live))
    @rule(data=st.data())
    def free_partial(self, data: st.DataObject) -> None:
        rid = data.draw(st.sampled_from(list(self.live)))
        ids = self.live[rid]
        if not ids:
            return
        keep_count = data.draw(st.integers(min_value=0, max_value=len(ids) - 1))
        to_free = ids[keep_count:]
        remaining = ids[:keep_count]
        prev_waste = self.pool._padding_waste_bytes
        self.pool.free(to_free)
        assert self.pool._padding_waste_bytes <= prev_waste
        if remaining:
            self.live[rid] = remaining
        else:
            self.live.pop(rid)

    @invariant()
    def capacity_conservation(self) -> None:
        table = self.pool._table
        assert table.num_pages_used + table.num_pages_free == table.num_pages_total

    @invariant()
    def lru_tracker_has_no_empty_entries(self) -> None:
        for entry in self.pool._tracker._entries.values():
            assert entry.page_ids, "tracker entry must never carry an empty page_ids list"

    @invariant()
    def padding_waste_is_non_negative(self) -> None:
        assert self.pool._padding_waste_bytes >= 0


TestPaddedUnifiedStateMachine = PaddedUnifiedStateMachine.TestCase
