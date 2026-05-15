"""Stateful property-based tests for :class:`PaddedUnifiedPool`.

Drives a single unified page table through allocate / free / partial-
free rules and asserts the documented baseline invariants:

- Pool capacity conservation: ``num_pages_used + num_pages_free ==
  num_pages_total`` at every step.
- ``padding_waste_bytes`` stays non-negative. The hardening PR
  originally claimed ``waste`` was monotonically non-decreasing during
  allocate; Hypothesis surfaced the counter-example (an allocate that
  internally triggers eviction frees evicted pages' waste before
  adding the new alloc's waste, and can net-decrease when the
  evicted-page count exceeds the new-alloc count). The user-facing
  invariant is just non-negativity, which matches the snapshot
  semantics ``benchmarks/README.md`` already documents.
- The LRU tracker carries no entry with an empty ``page_ids`` list
  (the 79a98b0 lesson, the same regression net the baseline tests
  pin).
- After free (no internal eviction), ``padding_waste_bytes`` cannot
  grow. This is a transition property; tracked per-rule.

The pool routes alloc/free identically regardless of layer kind, but
its waste accounting subtracts the per-kind logical size from the
unified page size, so both kinds are exercised.
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
