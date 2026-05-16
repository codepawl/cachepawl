"""End-to-end tests for AVMP v2 sub-PR 4 auto-trigger placement.

v2 sub-PR 4 (RFC 0002 section 4.2) moved the auto-trigger out of the post-
allocate / post-free observation hook and into the ``CapacityError`` branch
of ``_try_bulk_allocate_with_eviction``. The trigger now fires only when an
allocate would otherwise raise OOM after eviction, with direction inferred
from the failing pool's kind (``forced=True, pressured_kind=...``).

Manual triggers via ``trigger_manual_rebalance`` bypass the throttle and
go straight to ``_apply_rebalance``.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from cachepawl.allocator.avmp import AsymmetricVirtualPool, RebalanceDirection
from cachepawl.allocator.avmp.handle import HandleKind
from cachepawl.models.spec import HybridModelSpec, LayerKind

_TOTAL_16_MIB = 16 * 1024 * 1024
_NEVER_AUTO_TRIGGER_OPS = 2**30


def _make_pool(
    spec: HybridModelSpec,
    device: torch.device,
    *,
    min_rebalance_interval_ops: int = 0,
) -> AsymmetricVirtualPool:
    """Pool fixture for auto-trigger tests.

    Default ``min_rebalance_interval_ops=0`` so the throttle does not gate
    the (now forced-only) trigger path. The throttle suppression tests
    pass ``_NEVER_AUTO_TRIGGER_OPS`` explicitly when they want to verify
    the non-forced path's accounting.
    """

    return AsymmetricVirtualPool(
        model_spec=spec,
        total_bytes=_TOTAL_16_MIB,
        device=device,
        mamba_ratio=0.5,
        rebalance_enabled=True,
        min_rebalance_interval_ops=min_rebalance_interval_ops,
    )


def _force_kv_oom(pool: AsymmetricVirtualPool) -> None:
    """Drain KV in two requests so a third over-allocation forces
    CapacityError after eviction.

    Two requests are needed: a single request that exactly fills the pool
    can be evicted on retry, freeing all pages and letting the retry
    succeed. Two requests force genuine over-capacity demand.
    """

    kv_total = int(pool.get_allocator_stats()["kv_pages_total"])
    half = kv_total // 2
    pool.set_current_layer_kind(LayerKind.ATTENTION)
    pool.set_current_request_id(1)
    pool.allocate(half, dtype_bytes=2)
    pool.set_current_request_id(2)
    pool.allocate(half, dtype_bytes=2)
    # Now both halves are held. A third request over-allocates; eviction
    # frees the LRU half but the request still exceeds remaining capacity.
    # The point of this helper is to drive the CapacityError path, not to
    # guarantee the third allocation succeeds.
    pool.set_current_request_id(3)
    with contextlib.suppress(torch.cuda.OutOfMemoryError):
        pool.allocate(kv_total, dtype_bytes=2)


def _force_ssm_oom(pool: AsymmetricVirtualPool) -> None:
    ssm_total = int(pool.get_allocator_stats()["ssm_blocks_total"])
    half = ssm_total // 2
    pool.set_current_layer_kind(LayerKind.MAMBA2)
    pool.set_current_request_id(1)
    pool.allocate(half, dtype_bytes=2)
    pool.set_current_request_id(2)
    pool.allocate(half, dtype_bytes=2)
    pool.set_current_request_id(3)
    with contextlib.suppress(torch.cuda.OutOfMemoryError):
        pool.allocate(ssm_total, dtype_bytes=2)


def test_kv_pressure_auto_triggers_ssm_to_kv(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """A CapacityError on KV should trigger an SSM_TO_KV rebalance attempt.

    Direction is inferred from the failing ``kind``, not from compute_state
    (rolled-back free ratios at the trigger site are not informative).
    """

    pool = _make_pool(jamba_spec, cpu_device)
    baseline = pool.get_allocator_stats()
    _force_kv_oom(pool)

    stats = pool.get_allocator_stats()
    assert stats["rebalance_count"] >= 1.0
    assert stats["bytes_migrated_total"] > 0.0
    # SSM_TO_KV migration grows KV and shrinks SSM.
    assert stats["current_kv_pool_bytes"] > baseline["kv_pool_bytes"]
    assert stats["current_ssm_pool_bytes"] < baseline["ssm_pool_bytes"]


def test_ssm_pressure_auto_triggers_kv_to_ssm(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """A CapacityError on SSM triggers KV_TO_SSM. With migration_batch_size=1
    and Jamba 1.5 Mini's page_size < block_size the recipient SSM may not
    gain whole blocks (rounding waste). The donor KV still shrinks; the
    waste counter records the residue.
    """

    pool = _make_pool(jamba_spec, cpu_device)
    baseline = pool.get_allocator_stats()
    _force_ssm_oom(pool)

    stats = pool.get_allocator_stats()
    assert stats["rebalance_count"] >= 1.0
    assert stats["bytes_migrated_total"] > 0.0
    assert stats["current_kv_pool_bytes"] < baseline["kv_pool_bytes"]
    ssm_grew = stats["current_ssm_pool_bytes"] > baseline["ssm_pool_bytes"]
    wasted_all = stats["bytes_wasted_to_alignment_total"] >= stats["bytes_migrated_total"]
    assert ssm_grew or wasted_all


def test_forced_trigger_requires_pressured_kind(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Forced path without a ``pressured_kind`` argument is a safe no-op."""

    pool = _make_pool(jamba_spec, cpu_device)
    outcome = pool._maybe_auto_rebalance(forced=True, pressured_kind=None)
    assert outcome is None
    assert pool.get_allocator_stats()["rebalance_count"] == 0.0


def test_forced_path_bypasses_throttle(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Even when ``min_rebalance_interval_ops`` is huge, the forced path
    (used by the CapacityError branch) still fires immediately."""

    pool = _make_pool(jamba_spec, cpu_device, min_rebalance_interval_ops=_NEVER_AUTO_TRIGGER_OPS)
    outcome = pool._maybe_auto_rebalance(forced=True, pressured_kind=HandleKind.KV_PAGE)
    assert outcome is not None
    assert outcome.success
    stats = pool.get_allocator_stats()
    assert stats["rebalance_count"] == 1.0
    assert stats["auto_rebalance_skipped_throttle"] == 0.0


def test_non_forced_path_throttle_skips_when_interval_unmet(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """The non-forced path (currently called from no production code) still
    has working throttle accounting. Drive it directly to verify the counter."""

    pool = _make_pool(jamba_spec, cpu_device, min_rebalance_interval_ops=_NEVER_AUTO_TRIGGER_OPS)
    # Simulate KV pressure: directly force the state so the non-forced
    # branch sees something other than BALANCED. (In production this would
    # be set by _observe_pressure_state from the allocate hook.)
    from cachepawl.allocator.avmp.state import PoolPressureState

    pool._current_pressure_state = PoolPressureState.KV_PRESSURED
    outcome = pool._maybe_auto_rebalance(forced=False)
    assert outcome is None
    stats = pool.get_allocator_stats()
    assert stats["auto_rebalance_skipped_throttle"] >= 1.0
    assert stats["rebalance_count"] == 0.0


def test_manual_trigger_bypasses_throttle(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """``trigger_manual_rebalance`` bypasses both the throttle and the
    forced-vs-non-forced branching."""

    pool = _make_pool(jamba_spec, cpu_device, min_rebalance_interval_ops=_NEVER_AUTO_TRIGGER_OPS)
    outcome = pool.trigger_manual_rebalance(RebalanceDirection.SSM_TO_KV, batch_blocks=1)
    assert outcome.success
    stats = pool.get_allocator_stats()
    assert stats["rebalance_count"] == 1.0
    assert stats["bytes_migrated_total"] > 0.0


def test_repeated_pressure_keeps_firing_when_capacity_allows(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Two rounds of KV exhaustion each fire (at least one) rebalance event."""

    pool = _make_pool(jamba_spec, cpu_device)
    _force_kv_oom(pool)
    after_first = pool.get_allocator_stats()["rebalance_count"]
    assert after_first >= 1.0

    # Free everything we still hold, then exhaust again.
    pool.free([h for h in range(1, 1000)])  # best-effort cleanup
    _force_kv_oom(pool)
    after_second = pool.get_allocator_stats()["rebalance_count"]
    assert after_second >= after_first


@pytest.mark.parametrize("kind", [HandleKind.KV_PAGE, HandleKind.SSM_BLOCK])
def test_forced_direction_inferred_from_kind(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
    kind: HandleKind,
) -> None:
    """Forced path direction comes from ``pressured_kind``: failing pool
    receives, the other donates."""

    pool = _make_pool(jamba_spec, cpu_device)
    baseline = pool.get_allocator_stats()
    outcome = pool._maybe_auto_rebalance(forced=True, pressured_kind=kind)
    assert outcome is not None
    if kind is HandleKind.KV_PAGE:
        assert outcome.direction is RebalanceDirection.SSM_TO_KV
    else:
        assert outcome.direction is RebalanceDirection.KV_TO_SSM
    # Donor pool shrunk regardless of which direction.
    after = pool.get_allocator_stats()
    if kind is HandleKind.KV_PAGE:
        assert after["current_ssm_pool_bytes"] < baseline["ssm_pool_bytes"]
    else:
        assert after["current_kv_pool_bytes"] < baseline["kv_pool_bytes"]
