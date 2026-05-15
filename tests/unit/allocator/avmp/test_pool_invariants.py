"""Stress invariants for AsymmetricVirtualPool under eviction pressure.

Two long random sequences. The first asserts the four sanity
invariants from ``benchmarks/README.md`` after every 10 operations
on a tight 16 MiB budget where eviction is the norm rather than the
exception. The second tracks every minted handle id across the run
and asserts that no id is ever reused, which is the property
``VirtualPageTable`` promises but only the pool can exercise under
realistic allocate/evict pressure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.models.spec import HybridModelSpec, LayerKind

_TOTAL_16_MIB = 16 * 1024 * 1024


def _make_pool(spec: HybridModelSpec, device: torch.device) -> AsymmetricVirtualPool:
    return AsymmetricVirtualPool(
        model_spec=spec,
        total_bytes=_TOTAL_16_MIB,
        device=device,
        mamba_ratio=0.5,
    )


@dataclass(slots=True)
class _LiveRequest:
    request_id: int
    kind: LayerKind
    ids: list[int]


def _assert_sanity_invariants(pool: AsymmetricVirtualPool, kv_total: int, ssm_total: int) -> None:
    stats = pool.get_allocator_stats()
    assert stats["kv_pages_used"] + stats["kv_pages_free"] == float(kv_total)
    assert stats["ssm_blocks_used"] + stats["ssm_blocks_free"] == float(ssm_total)
    assert stats["virtual_handles_live"] == stats["kv_pages_used"] + stats["ssm_blocks_used"]
    assert stats["cross_pool_eviction_count"] == 0.0


def test_500_random_ops_with_eviction_pressure(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    pool = _make_pool(jamba_spec, cpu_device)
    initial = pool.get_allocator_stats()
    kv_total = int(initial["kv_pages_total"])
    ssm_total = int(initial["ssm_blocks_total"])

    rng = random.Random(42)
    live: list[_LiveRequest] = []
    next_request_id = 1

    for step in range(500):
        roll = rng.random()
        if live and roll < 0.35:
            victim = live.pop(rng.randrange(len(live)))
            pool.free(victim.ids)
        else:
            kind = LayerKind.ATTENTION if rng.random() < 0.6 else LayerKind.MAMBA2
            pool.set_current_layer_kind(kind)
            pool.set_current_request_id(next_request_id)
            count = rng.randint(1, 6)
            try:
                ids = pool.allocate(count, dtype_bytes=2)
            except torch.cuda.OutOfMemoryError:
                ids = []
            if ids:
                live.append(_LiveRequest(request_id=next_request_id, kind=kind, ids=ids))
            next_request_id += 1

        if step % 10 == 0:
            _assert_sanity_invariants(pool, kv_total, ssm_total)

    _assert_sanity_invariants(pool, kv_total, ssm_total)


def test_handle_ids_are_never_reused_under_pressure(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Stress the page table's non-reuse invariant via the pool.

    PR1's :class:`VirtualPageTable` promises monotonic, non-reusing
    handle ids; the pool is the realistic surface that exercises this
    under alloc + evict + free pressure. A duplicate id over the whole
    run would mean the page table started reusing, which would silently
    re-enable the use-after-free class of bug AVMP is designed to
    refuse.
    """

    pool = _make_pool(jamba_spec, cpu_device)
    rng = random.Random(123)
    live: list[_LiveRequest] = []
    seen_ids: set[int] = set()
    next_request_id = 1

    for _ in range(300):
        roll = rng.random()
        if live and roll < 0.35:
            victim = live.pop(rng.randrange(len(live)))
            pool.free(victim.ids)
            continue
        kind = LayerKind.ATTENTION if rng.random() < 0.6 else LayerKind.MAMBA2
        pool.set_current_layer_kind(kind)
        pool.set_current_request_id(next_request_id)
        count = rng.randint(1, 5)
        try:
            ids = pool.allocate(count, dtype_bytes=2)
        except torch.cuda.OutOfMemoryError:
            ids = []
        for hid in ids:
            assert hid not in seen_ids, f"handle id {hid} was reused"
            seen_ids.add(hid)
        if ids:
            live.append(_LiveRequest(request_id=next_request_id, kind=kind, ids=ids))
        next_request_id += 1
