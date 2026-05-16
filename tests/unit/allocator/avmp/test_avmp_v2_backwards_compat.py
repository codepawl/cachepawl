"""v1 byte-identity contract for AVMP v2.

When ``rebalance_enabled=False`` (the constructor default through sub-PR
3) the pool reproduces v1 behavior: the same allocate / free sequence
produces the same ``get_allocator_stats`` snapshot regardless of which
new v2 kwargs are passed. Sub-PR 3's new ``min_rebalance_interval_ops``
param does not change behavior when no auto-trigger or manual trigger
fires.

The deterministic-subset contract from RFC 0002 section 2 lives here.
"""

from __future__ import annotations

import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool
from cachepawl.models.spec import HybridModelSpec, LayerKind

_TOTAL_16_MIB = 16 * 1024 * 1024

# Timing-sensitive fields skipped when comparing stats dicts; the
# allocator's behavior is deterministic but the wall clock is not.
_TIMING_FIELDS: frozenset[str] = frozenset({"time_spent_rebalancing_ns"})


def _drive(pool: AsymmetricVirtualPool) -> None:
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    pool.allocate(5, dtype_bytes=2)
    pool.set_current_layer_kind(LayerKind.MAMBA2)
    pool.set_current_request_id(2)
    pool.allocate(2, dtype_bytes=2)
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(3)
    pool.allocate(3, dtype_bytes=2)


def _make_pool(
    spec: HybridModelSpec,
    device: torch.device,
    *,
    rebalance_enabled: bool,
    min_rebalance_interval_ops: int = 1000,
) -> AsymmetricVirtualPool:
    return AsymmetricVirtualPool(
        model_spec=spec,
        total_bytes=_TOTAL_16_MIB,
        device=device,
        mamba_ratio=0.5,
        rebalance_enabled=rebalance_enabled,
        min_rebalance_interval_ops=min_rebalance_interval_ops,
    )


def test_rebalance_disabled_with_v2_kwargs_matches_v1_baseline(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Two pools, identical config except one passes the new
    ``min_rebalance_interval_ops`` kwarg, produce byte-identical stats."""

    pool_default = _make_pool(jamba_spec, cpu_device, rebalance_enabled=False)
    pool_with_v2_kwargs = _make_pool(
        jamba_spec, cpu_device, rebalance_enabled=False, min_rebalance_interval_ops=5000
    )

    _drive(pool_default)
    _drive(pool_with_v2_kwargs)

    stats_default = dict(pool_default.get_allocator_stats())
    stats_v2 = dict(pool_with_v2_kwargs.get_allocator_stats())
    for key in _TIMING_FIELDS:
        stats_default.pop(key, None)
        stats_v2.pop(key, None)
    assert stats_default == stats_v2


def test_rebalance_disabled_runs_no_migration_under_pressure(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """With ``rebalance_enabled=False`` neither the auto-trigger nor the
    monitor exists; capacity stays static, counters stay zero."""

    pool = _make_pool(jamba_spec, cpu_device, rebalance_enabled=False)
    baseline = pool.get_allocator_stats()
    kv_total = int(baseline["kv_pages_total"])

    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    pool.allocate(kv_total, dtype_bytes=2)

    stats = pool.get_allocator_stats()
    assert stats["current_kv_pool_bytes"] == baseline["kv_pool_bytes"]
    assert stats["current_ssm_pool_bytes"] == baseline["ssm_pool_bytes"]
    assert stats["rebalance_count"] == 0.0
    assert stats["bytes_migrated_total"] == 0.0
    assert stats["time_spent_rebalancing_ns"] == 0.0
    assert stats["auto_rebalance_skipped_throttle"] == 0.0
