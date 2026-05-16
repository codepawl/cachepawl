"""Stateful property-based tests for :class:`AsymmetricVirtualPool`.

Drives the allocator through random sequences of allocate / free /
partial-free rules and asserts AVMP's three documented invariants
after every step plus a fourth invariant on the LRU trackers (no
empty entries linger after :meth:`remove_pages` runs).

A failure here is a real bug. The seed is pinned by the stateful
profile in ``conftest.py`` so any failure reproduces from the test
log alone; the minimal counter-example should be added to
``tests/unit/allocator/avmp/test_pool_basic.py`` (or a neighbor) as a
seed-based regression test before fixing the allocator.
"""

from __future__ import annotations

import hypothesis.strategies as st
import torch
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.models.spec import JAMBA_1_5_MINI_REF, LayerKind

_TOTAL_4_MIB = 4 * 1024 * 1024


class AvmpStateMachine(RuleBasedStateMachine):
    """Random allocate/free workload against AVMP with invariant checks."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = AsymmetricVirtualPool(
            model_spec=JAMBA_1_5_MINI_REF,
            total_bytes=_TOTAL_4_MIB,
            device=torch.device("cpu"),
            mamba_ratio=0.5,
        )
        self.live_kv: dict[int, list[int]] = {}
        self.live_ssm: dict[int, list[int]] = {}
        self.next_request_id = 1

    @rule(count=st.integers(min_value=1, max_value=8))
    def allocate_kv(self, count: int) -> None:
        self.pool.set_current_layer_kind(LayerKind.ATTENTION)
        self.pool.set_current_request_id(self.next_request_id)
        try:
            ids = self.pool.allocate(count, dtype_bytes=2)
        except torch.cuda.OutOfMemoryError:
            self.next_request_id += 1
            return
        if ids:
            self.live_kv[self.next_request_id] = ids
        self.next_request_id += 1

    @rule()
    def allocate_ssm(self) -> None:
        self.pool.set_current_layer_kind(LayerKind.MAMBA2)
        self.pool.set_current_request_id(self.next_request_id)
        try:
            ids = self.pool.allocate(1, dtype_bytes=2)
        except torch.cuda.OutOfMemoryError:
            self.next_request_id += 1
            return
        if ids:
            self.live_ssm[self.next_request_id] = ids
        self.next_request_id += 1

    @precondition(lambda self: bool(self.live_kv) or bool(self.live_ssm))
    @rule(data=st.data())
    def free_request(self, data: st.DataObject) -> None:
        choices = list(self.live_kv) + list(self.live_ssm)
        rid = data.draw(st.sampled_from(choices))
        ids = self.live_kv.pop(rid) if rid in self.live_kv else self.live_ssm.pop(rid)
        self.pool.free(ids)

    @precondition(lambda self: bool(self.live_kv) or bool(self.live_ssm))
    @rule(data=st.data())
    def free_partial(self, data: st.DataObject) -> None:
        choices = list(self.live_kv) + list(self.live_ssm)
        rid = data.draw(st.sampled_from(choices))
        live_map = self.live_kv if rid in self.live_kv else self.live_ssm
        ids = live_map[rid]
        if not ids:
            return
        keep_count = data.draw(st.integers(min_value=0, max_value=len(ids) - 1))
        # Free everything past keep_count; if keep_count == 0 this is a full free.
        to_free = ids[keep_count:]
        remaining = ids[:keep_count]
        self.pool.free(to_free)
        if remaining:
            live_map[rid] = remaining
        else:
            live_map.pop(rid)

    @invariant()
    def capacity_conservation(self) -> None:
        stats = self.pool.get_allocator_stats()
        assert stats["kv_pages_used"] + stats["kv_pages_free"] == stats["kv_pages_total"]
        assert stats["ssm_blocks_used"] + stats["ssm_blocks_free"] == stats["ssm_blocks_total"]

    @invariant()
    def virtual_handles_match_used_counts(self) -> None:
        stats = self.pool.get_allocator_stats()
        assert stats["virtual_handles_live"] == stats["kv_pages_used"] + stats["ssm_blocks_used"]

    @invariant()
    def cross_pool_eviction_count_stays_zero_in_v1(self) -> None:
        stats = self.pool.get_allocator_stats()
        assert stats["cross_pool_eviction_count"] == 0.0

    @invariant()
    def lru_trackers_have_no_empty_entries(self) -> None:
        for tracker in (self.pool._kv_tracker, self.pool._ssm_tracker):
            for entry in tracker._entries.values():
                assert entry.page_ids, "tracker entry must never carry an empty page_ids list"


TestAvmpStateMachine = AvmpStateMachine.TestCase
