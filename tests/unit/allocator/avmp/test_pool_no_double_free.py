"""Regression tests for the eviction-induced double-free class of bug.

In commit 79a98b0 the baseline allocators were caught not calling
``LRURequestTracker.remove_pages`` on the public free path. A
subsequent ``select_oldest`` would return a request whose pages were
already freed, ``drop`` would return its stale page ids, and the
eviction code would re-free them, growing ``num_pages_free`` past
``num_pages_total`` and silently corrupting the pool.

AVMP's ``free`` is wired the same way the fixed baselines now are
(``remove_pages`` after the store-level frees). These tests assert
the invariant from the outside, so a future regression in either the
pool or the tracker shows up here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.models.spec import HybridModelSpec, LayerKind

_TOTAL_4_MIB = 4 * 1024 * 1024


def _make_pool(spec: HybridModelSpec, device: torch.device) -> AsymmetricVirtualPool:
    return AsymmetricVirtualPool(
        model_spec=spec,
        total_bytes=_TOTAL_4_MIB,
        device=device,
        mamba_ratio=0.5,
    )


def test_eviction_after_clean_departure_does_not_double_free(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Request A departs cleanly; later eviction must skip A and pick B.

    Numbers chosen so that the only way C can succeed is if eviction
    correctly picks B (not the already-freed A). With the 79a98b0 bug
    in place, select_oldest would return A, drop would yield A's old
    handle ids, and the page-table miss would leave the pool unable
    to satisfy C.
    """

    pool = _make_pool(jamba_spec, cpu_device)
    kv_total = int(pool.get_allocator_stats()["kv_pages_total"])
    assert kv_total >= 4  # sanity: the math below assumes >= 4 KV pages

    # Request A: take a chunk, then depart cleanly.
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    a_ids = pool.allocate(kv_total - 2, dtype_bytes=2)
    pool.free(a_ids)

    # Request B: take the same shape, leaving 2 pages free.
    pool.set_current_request_id(2)
    b_ids = pool.allocate(kv_total - 2, dtype_bytes=2)

    # Request C: demand more than 2; forces one eviction. The fix
    # case evicts B (the oldest live request) and yields kv_total
    # pages free, which satisfies the request. The bug case picks A
    # (stale tracker entry), gets nothing back, and OOMs.
    pool.set_current_request_id(3)
    c_ids = pool.allocate(3, dtype_bytes=2)
    assert len(c_ids) == 3

    stats = pool.get_allocator_stats()
    # Only C is live; B was evicted, A had departed.
    assert stats["kv_pages_used"] == 3.0
    assert stats["virtual_handles_live"] == 3.0
    assert stats["kv_pages_free"] == kv_total - 3

    # B's handles must no longer resolve.
    pool.free(b_ids)  # idempotent on already-evicted ids
    assert pool.get_allocator_stats()["kv_pages_used"] == 3.0


def test_repeated_clean_departure_and_realloc_stays_consistent(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """A cycle of alloc/free for the same request id must not leak.

    Without ``remove_pages`` the tracker would accumulate one entry
    per cycle even though each request id departs cleanly, and
    ``select_oldest`` would eventually pick a stale entry.
    """

    pool = _make_pool(jamba_spec, cpu_device)
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    for cycle in range(8):
        pool.set_current_request_id(cycle)
        ids = pool.allocate(2, dtype_bytes=2)
        pool.free(ids)
        stats = pool.get_allocator_stats()
        assert stats["kv_pages_used"] == 0.0
        assert stats["virtual_handles_live"] == 0.0


@dataclass(slots=True)
class _LiveRequest:
    request_id: int
    kind: LayerKind
    ids: list[int]


def test_stress_random_ops_invariants(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """200 random allocate/free ops keep the per-store occupancy invariant."""

    pool = _make_pool(jamba_spec, cpu_device)
    kv_total = int(pool.get_allocator_stats()["kv_pages_total"])
    ssm_total = int(pool.get_allocator_stats()["ssm_blocks_total"])

    rng = random.Random(42)
    live: list[_LiveRequest] = []
    next_request_id = 100

    for _ in range(200):
        roll = rng.random()
        if live and roll < 0.4:
            # Free a random live request.
            victim = live.pop(rng.randrange(len(live)))
            pool.free(victim.ids)
        else:
            # Allocate. Pick KV or SSM, a small count, a fresh request id.
            kind = LayerKind.ATTENTION if rng.random() < 0.6 else LayerKind.MAMBA2
            pool.set_current_layer_kind(kind)
            pool.set_current_request_id(next_request_id)
            count = rng.randint(1, 4)
            try:
                ids = pool.allocate(count, dtype_bytes=2)
            except torch.cuda.OutOfMemoryError:
                # OOM is allowed; the invariant still holds and we
                # move on to the next op.
                continue
            if ids:
                live.append(_LiveRequest(request_id=next_request_id, kind=kind, ids=ids))
            next_request_id += 1

        stats = pool.get_allocator_stats()
        assert stats["kv_pages_used"] + stats["kv_pages_free"] == float(kv_total)
        assert stats["ssm_blocks_used"] + stats["ssm_blocks_free"] == float(ssm_total)
        assert stats["virtual_handles_live"] >= 0.0
        assert stats["kv_pages_used"] <= float(kv_total)
        assert stats["ssm_blocks_used"] <= float(ssm_total)
