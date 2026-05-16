"""AsymmetricVirtualPool: AVMP v1 allocator.

Composes the PR1 primitives (``KVPagesStore``, ``SSMBlocksStore``,
``VirtualPageTable``) behind the :class:`Allocator` ABC, mirroring
:class:`FixedDualPool` for API consistency. The single distinct
AVMP behavior in v1 is the virtual handle abstraction: the integer
ids returned by :meth:`allocate` are minted by ``VirtualPageTable``
and never reused within one pool lifetime, so a stale handle's
``free`` call is rejected by :meth:`VirtualPageTable.remove`
rather than silently colliding with an unrelated fresh allocation.

Scope of v1, deliberately scaffolded:

- Static partition at construction time, identical formula to
  ``FixedDualPool``. ``mamba_ratio`` selects the fraction of
  ``total_bytes`` assigned to the SSM pool.
- Per-pool LRU eviction with cross-pool isolation. Filling the KV
  pool while SSM has free bytes raises
  :class:`torch.cuda.OutOfMemoryError`, not an eviction into the
  SSM pool. The ``cross_pool_eviction_count`` stat exists and
  always reports ``0.0`` in v1; the field stays in the schema so
  the v2 cross-pool migration path can light it up without a
  follow-on schema bump.

Performance expectation versus :class:`FixedDualPool` at the same
``mamba_ratio``: parity. The contribution that beats ``fixed_dual``
on rigidity is dynamic cross-pool rebalancing per
``docs/designs/0001-asymmetric-virtual-memory-paging.md`` section
4.5, which lands in a follow-up PR.

PR seam notes:

- ``VirtualHandle.layer_idx`` is recorded as ``0`` for every handle.
  :class:`AllocatorContext` only carries layer kind, not layer
  index. A future extension can carry the index if real per-layer
  accounting becomes useful; for v1 the field is a placeholder.
- ``VirtualHandle.virtual_offset`` equals the physical offset in v1
  because no remapping happens. Both numbers will diverge in a
  later milestone when cross-pool migration is introduced and the
  virtual offset becomes stable across physical relocation.

Primitives reused, with file paths:

- ``src/cachepawl/allocator/avmp/physical.py``: ``KVPagesStore`` and
  ``SSMBlocksStore``.
- ``src/cachepawl/allocator/avmp/page_table.py``: ``VirtualPageTable``.
- ``src/cachepawl/allocator/baselines/common.py``: ``LRURequestTracker``,
  ``AllocatorContext``, ``CapacityError``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import torch

from cachepawl.allocator.avmp.handle import HandleKind
from cachepawl.allocator.avmp.page_table import VirtualPageTable
from cachepawl.allocator.avmp.physical import KVPagesStore, SSMBlocksStore
from cachepawl.allocator.avmp.state import (
    PoolPressureMonitor,
    PoolPressureState,
    RebalanceDirection,
    RebalanceOutcome,
)
from cachepawl.allocator.base import Allocator, AllocatorStats
from cachepawl.allocator.baselines.common import (
    AllocatorContext,
    CapacityError,
    LRURequestTracker,
)
from cachepawl.allocator.policy import EvictionPolicy
from cachepawl.models.spec import HybridModelSpec, LayerKind


class AsymmetricVirtualPool(Allocator, AllocatorContext):
    """Static-partition AVMP allocator with same-pool LRU eviction."""

    def __init__(
        self,
        model_spec: HybridModelSpec,
        total_bytes: int,
        device: torch.device,
        mamba_ratio: float = 0.5,
        attention_page_tokens: int = 16,
        eviction: EvictionPolicy = EvictionPolicy.LRU,
        rebalance_enabled: bool = False,
        threshold_low: float = 0.05,
        threshold_high: float = 0.30,
        migration_batch_size: int = 1,
        min_rebalance_interval_ns: int = 1_000_000,
    ) -> None:
        if eviction is not EvictionPolicy.LRU:
            raise NotImplementedError(
                f"AsymmetricVirtualPool: eviction={eviction.name} is not implemented in v1; "
                "only EvictionPolicy.LRU is wired up."
            )
        if total_bytes <= 0:
            raise ValueError(f"total_bytes must be positive, got {total_bytes}")
        if not 0.0 < mamba_ratio < 1.0:
            raise ValueError(f"mamba_ratio must lie in the open interval (0, 1), got {mamba_ratio}")
        if attention_page_tokens <= 0:
            raise ValueError(f"attention_page_tokens must be positive, got {attention_page_tokens}")
        if not 0.0 < threshold_low < threshold_high < 1.0:
            raise ValueError(
                "thresholds must satisfy 0.0 < threshold_low < threshold_high < 1.0, got "
                f"threshold_low={threshold_low}, threshold_high={threshold_high}"
            )
        if migration_batch_size < 1:
            raise ValueError(f"migration_batch_size must be >= 1, got {migration_batch_size}")
        if min_rebalance_interval_ns < 0:
            raise ValueError(
                f"min_rebalance_interval_ns must be >= 0, got {min_rebalance_interval_ns}"
            )

        AllocatorContext.__init__(self)
        self._model_spec = model_spec
        self._mamba_ratio = mamba_ratio
        self._rebalance_enabled = rebalance_enabled
        self._threshold_low = threshold_low
        self._threshold_high = threshold_high
        self._migration_batch_size = migration_batch_size
        self._min_rebalance_interval_ns = min_rebalance_interval_ns

        ssm_bytes = int(total_bytes * mamba_ratio)
        kv_bytes = total_bytes - ssm_bytes
        if kv_bytes <= 0 or ssm_bytes <= 0:
            raise ValueError(
                f"computed per-pool size invalid: kv_bytes={kv_bytes} ssm_bytes={ssm_bytes}"
            )

        # v2 sub-PR 2 (RFC 0002 section 4.3): pre-allocate each BackingStore
        # at pool.total_bytes so that resize_capacity can grow without a
        # torch tensor reallocation. The active capacity is capped to the
        # per-store share via initial_capacity_bytes. Physical RAM footprint
        # is 2 * total_bytes in the Python prototype; the future CUDA
        # implementation will use cuMemMap (RFC 0002 section 4.5).
        self._kv_store = KVPagesStore(
            model_spec=model_spec,
            attention_page_tokens=attention_page_tokens,
            total_bytes=total_bytes,
            device=device,
            initial_capacity_bytes=kv_bytes,
        )
        self._ssm_store = SSMBlocksStore(
            model_spec=model_spec,
            total_bytes=total_bytes,
            device=device,
            initial_capacity_bytes=ssm_bytes,
        )
        self._page_table = VirtualPageTable()
        self._kv_tracker = LRURequestTracker()
        self._ssm_tracker = LRURequestTracker()
        self._cross_pool_eviction_count: int = 0

        self._pressure_monitor: PoolPressureMonitor | None = (
            PoolPressureMonitor(
                threshold_low=threshold_low,
                threshold_high=threshold_high,
            )
            if rebalance_enabled
            else None
        )
        self._current_pressure_state: PoolPressureState = PoolPressureState.BALANCED

        # v2 sub-PR 2 (RFC 0002 section 4.7) migration counters.
        self._rebalance_count: int = 0
        self._bytes_migrated_total: int = 0
        self._bytes_wasted_to_alignment_total: int = 0
        self._time_spent_rebalancing_ns: int = 0
        # v2 sub-PR 3 (RFC 0002 section 4.2, section 8 question 3) auto-trigger throttle.
        self._last_auto_rebalance_ns: int = 0
        self._auto_rebalance_skipped_throttle: int = 0

    # ----- Allocator ABC -----

    def allocate(self, num_blocks: int, *, dtype_bytes: int) -> list[int]:
        if num_blocks <= 0:
            return []
        if dtype_bytes <= 0:
            raise ValueError(f"dtype_bytes must be positive, got {dtype_bytes}")
        kind = self._kind_from_context()
        result = self._allocate_into(kind=kind, num_blocks=num_blocks)
        # v2 sub-PR 3: observe state and (when rebalance_enabled) attempt an
        # auto-rebalance if the pool is pressured (RFC 0002 section 4.2).
        self._observe_pressure_state()
        return result

    def free(self, block_ids: Sequence[int]) -> None:
        seen: set[int] = set()
        kv_freed: list[int] = []
        ssm_freed: list[int] = []
        for hid in block_ids:
            if hid in seen:
                continue
            seen.add(hid)
            try:
                handle, physical_offset = self._page_table.remove(hid)
            except KeyError:
                continue
            if handle.kind is HandleKind.KV_PAGE:
                self._kv_store.free_one(physical_offset)
                kv_freed.append(hid)
            else:
                self._ssm_store.free_one(physical_offset)
                ssm_freed.append(hid)
        if kv_freed:
            self._kv_tracker.remove_pages(kv_freed)
        if ssm_freed:
            self._ssm_tracker.remove_pages(ssm_freed)
        # v2 sub-PR 3: observe state and (when rebalance_enabled) attempt an
        # auto-rebalance if the pool is pressured (RFC 0002 section 4.2).
        self._observe_pressure_state()

    def stats(self) -> AllocatorStats:
        kv_total = self._kv_store.num_total
        ssm_total = self._ssm_store.num_total
        kv_used = self._kv_store.num_used
        ssm_used = self._ssm_store.num_used
        total = kv_total + ssm_total
        allocated = kv_used + ssm_used
        free = total - allocated
        ratio = (free / total) if total > 0 else 0.0
        return AllocatorStats(
            total_blocks=total,
            free_blocks=free,
            allocated_blocks=allocated,
            fragmentation_ratio=ratio,
        )

    # ----- Allocator-specific stats surface (duck-typed by the runner) -----

    def get_allocator_stats(self) -> Mapping[str, float]:
        """Return the AVMP stats dict.

        The 11 v1 keys are unchanged. The 14 v2 keys implement RFC 0002
        section 4.7's observability surface (12 from sub-PR 1, plus
        ``bytes_wasted_to_alignment_total`` from sub-PR 2, plus
        ``auto_rebalance_skipped_throttle`` from sub-PR 3).

        ``current_pressure_state_code`` encodes the :class:`PoolPressureState`
        enum: BALANCED=0, KV_PRESSURED=1, SSM_PRESSURED=2, REBALANCING=3.

        ``current_kv_pool_bytes`` and ``current_ssm_pool_bytes`` reflect the
        live per-store active capacity. They equal their v1 counterparts
        (``kv_pool_bytes`` / ``ssm_pool_bytes``) until the first successful
        rebalance.

        ``rebalance_count``, ``bytes_migrated_total``,
        ``bytes_wasted_to_alignment_total``, and ``time_spent_rebalancing_ns``
        accumulate across successful migrations.

        ``auto_rebalance_skipped_throttle`` counts the number of times the
        observation hook detected pressure but the throttle window had not
        elapsed since the previous auto-trigger.
        """

        kv_page_size = self._kv_store.page_size_bytes
        ssm_block_size = self._ssm_store.block_size_bytes
        kv_pool_bytes = self._kv_store.num_total * kv_page_size
        ssm_pool_bytes = self._ssm_store.num_total * ssm_block_size
        kv_free_ratio, ssm_free_ratio = self._free_ratios()
        return {
            # v1 keys (unchanged)
            "kv_pages_total": float(self._kv_store.num_total),
            "kv_pages_used": float(self._kv_store.num_used),
            "kv_pages_free": float(self._kv_store.num_free),
            "ssm_blocks_total": float(self._ssm_store.num_total),
            "ssm_blocks_used": float(self._ssm_store.num_used),
            "ssm_blocks_free": float(self._ssm_store.num_free),
            "virtual_handles_live": float(
                self._page_table.num_kv_handles_live + self._page_table.num_ssm_handles_live
            ),
            "cross_pool_eviction_count": float(self._cross_pool_eviction_count),
            "kv_pool_bytes": float(kv_pool_bytes),
            "ssm_pool_bytes": float(ssm_pool_bytes),
            "mamba_ratio": float(self._mamba_ratio),
            # v2 observability keys (RFC 0002 section 4.7)
            "rebalance_enabled": 1.0 if self._rebalance_enabled else 0.0,
            "threshold_low": float(self._threshold_low),
            "threshold_high": float(self._threshold_high),
            "migration_batch_size": float(self._migration_batch_size),
            "current_kv_pool_bytes": float(kv_pool_bytes),
            "current_ssm_pool_bytes": float(ssm_pool_bytes),
            "kv_free_ratio": float(kv_free_ratio),
            "ssm_free_ratio": float(ssm_free_ratio),
            "current_pressure_state_code": float(self._current_pressure_state.value),
            "rebalance_count": float(self._rebalance_count),
            "bytes_migrated_total": float(self._bytes_migrated_total),
            "time_spent_rebalancing_ns": float(self._time_spent_rebalancing_ns),
            "bytes_wasted_to_alignment_total": float(self._bytes_wasted_to_alignment_total),
            "auto_rebalance_skipped_throttle": float(self._auto_rebalance_skipped_throttle),
        }

    # ----- Helpers -----

    def _kind_from_context(self) -> HandleKind:
        if self._current_layer_kind is LayerKind.ATTENTION:
            return HandleKind.KV_PAGE
        return HandleKind.SSM_BLOCK

    def _allocate_into(self, *, kind: HandleKind, num_blocks: int) -> list[int]:
        offsets = self._try_bulk_allocate_with_eviction(kind=kind, num_blocks=num_blocks)
        size_bytes = self._size_bytes_for(kind)
        handle_ids: list[int] = []
        request_id_str = str(self._current_request_id)
        for offset in offsets:
            handle = self._page_table.mint(
                kind=kind,
                virtual_offset=offset,
                size_bytes=size_bytes,
                request_id=request_id_str,
                layer_idx=0,
                physical_offset=offset,
            )
            handle_ids.append(handle.handle_id)
        tracker = self._tracker_for(kind)
        tracker.touch(self._current_request_id, handle_ids)
        return handle_ids

    def _try_bulk_allocate_with_eviction(self, *, kind: HandleKind, num_blocks: int) -> list[int]:
        try:
            return self._bulk_allocate(kind=kind, num_blocks=num_blocks)
        except CapacityError as first:
            self._evict_one(kind=kind)
            try:
                return self._bulk_allocate(kind=kind, num_blocks=num_blocks)
            except CapacityError as exc:
                raise torch.cuda.OutOfMemoryError(
                    f"avmp pool ({kind.name}) exhausted after eviction (origin={first}): {exc}"
                ) from exc

    def _bulk_allocate(self, *, kind: HandleKind, num_blocks: int) -> list[int]:
        store_alloc = (
            self._kv_store.allocate_one
            if kind is HandleKind.KV_PAGE
            else self._ssm_store.allocate_one
        )
        store_free = (
            self._kv_store.free_one if kind is HandleKind.KV_PAGE else self._ssm_store.free_one
        )
        acquired: list[int] = []
        try:
            for _ in range(num_blocks):
                acquired.append(store_alloc())
        except CapacityError:
            for offset in acquired:
                store_free(offset)
            raise
        return acquired

    def _evict_one(self, *, kind: HandleKind) -> None:
        tracker = self._tracker_for(kind)
        victim_id = tracker.select_oldest()
        if victim_id is None:
            return
        victim_handle_ids = tracker.drop(victim_id)
        for hid in victim_handle_ids:
            try:
                handle, physical_offset = self._page_table.remove(hid)
            except KeyError:
                continue
            if handle.kind is HandleKind.KV_PAGE:
                self._kv_store.free_one(physical_offset)
            else:
                self._ssm_store.free_one(physical_offset)

    def _tracker_for(self, kind: HandleKind) -> LRURequestTracker:
        if kind is HandleKind.KV_PAGE:
            return self._kv_tracker
        return self._ssm_tracker

    def _size_bytes_for(self, kind: HandleKind) -> int:
        if kind is HandleKind.KV_PAGE:
            return self._kv_store.page_size_bytes
        return self._ssm_store.block_size_bytes

    def _free_ratios(self) -> tuple[float, float]:
        """Per-pool free fractions used by the pressure monitor and v2 stats."""

        kv_total = self._kv_store.num_total
        ssm_total = self._ssm_store.num_total
        kv_free_ratio = self._kv_store.num_free / kv_total if kv_total > 0 else 0.0
        ssm_free_ratio = self._ssm_store.num_free / ssm_total if ssm_total > 0 else 0.0
        return kv_free_ratio, ssm_free_ratio

    def _observe_pressure_state(self) -> None:
        """Read pressure state, record transitions, and attempt auto-rebalance.

        Two-step:

        1. Compute the new pressure state via the monitor and log any
           transition into the ring buffer.
        2. v2 sub-PR 3: when the new state is ``KV_PRESSURED`` or
           ``SSM_PRESSURED`` AND the per-pool throttle interval has elapsed,
           call ``_apply_rebalance`` to relieve the pressure before returning
           to the caller.

        When ``rebalance_enabled=False`` the monitor is ``None`` and both
        steps no-op.
        """

        monitor = self._pressure_monitor
        if monitor is None:
            return
        kv_free_ratio, ssm_free_ratio = self._free_ratios()
        new_state = monitor.compute_state(kv_free_ratio, ssm_free_ratio)
        if new_state is not self._current_pressure_state:
            monitor.record_transition(
                self._current_pressure_state,
                new_state,
                time.monotonic_ns(),
            )
            self._current_pressure_state = new_state
        self._maybe_auto_rebalance()

    def _maybe_auto_rebalance(self) -> RebalanceOutcome | None:
        """Trigger ``_apply_rebalance`` if the pool is pressured and the
        throttle window has elapsed.

        No-op when:

        - the monitor is absent (``rebalance_enabled=False``);
        - the current state is ``BALANCED`` or ``REBALANCING``;
        - less than ``min_rebalance_interval_ns`` has elapsed since the
          previous auto-trigger (increments ``auto_rebalance_skipped_throttle``).

        Manual triggers via ``trigger_manual_rebalance`` DO NOT consult the
        throttle; it exists only to bound the auto-trigger rate under fast-
        alternating workloads (RFC 0002 section 8 question 3).
        """

        monitor = self._pressure_monitor
        if monitor is None:
            return None
        state = self._current_pressure_state
        if state is PoolPressureState.BALANCED or state is PoolPressureState.REBALANCING:
            return None
        now_ns = time.monotonic_ns()
        if now_ns - self._last_auto_rebalance_ns < self._min_rebalance_interval_ns:
            self._auto_rebalance_skipped_throttle += 1
            return None
        direction = (
            RebalanceDirection.SSM_TO_KV
            if state is PoolPressureState.KV_PRESSURED
            else RebalanceDirection.KV_TO_SSM
        )
        self._last_auto_rebalance_ns = now_ns
        return self._apply_rebalance(direction, self._migration_batch_size)

    # ----- v2 sub-PR 2: migration mechanics (RFC 0002 section 4.3) -----

    def trigger_manual_rebalance(
        self,
        direction: RebalanceDirection,
        batch_blocks: int,
    ) -> RebalanceOutcome:
        """Diagnostic / test-only manual rebalance trigger.

        Sub-PR 3 wires the automatic trigger off the pressure monitor; this
        method remains thereafter for diagnostic and test purposes.

        Allowed regardless of ``rebalance_enabled``. When
        ``rebalance_enabled=False`` the pressure monitor is absent, so
        REBALANCING-state transitions are NOT recorded in the ring buffer;
        the migration counters in ``get_allocator_stats`` still update.
        """

        if batch_blocks < 1:
            raise ValueError(f"batch_blocks must be >= 1, got {batch_blocks}")
        return self._apply_rebalance(direction, batch_blocks)

    def _apply_rebalance(
        self,
        direction: RebalanceDirection,
        batch_blocks: int,
    ) -> RebalanceOutcome:
        start_ns = time.monotonic_ns()
        kv_page_size = self._kv_store.page_size_bytes
        ssm_block_size = self._ssm_store.block_size_bytes
        kv_bytes_before = self._kv_store.num_total * kv_page_size
        ssm_bytes_before = self._ssm_store.num_total * ssm_block_size

        if direction is RebalanceDirection.SSM_TO_KV:
            donor_store: KVPagesStore | SSMBlocksStore = self._ssm_store
            donor_unit = ssm_block_size
            donor_bytes_before = ssm_bytes_before
            recipient_store: KVPagesStore | SSMBlocksStore = self._kv_store
            recipient_unit = kv_page_size
            recipient_bytes_before = kv_bytes_before
        else:
            donor_store = self._kv_store
            donor_unit = kv_page_size
            donor_bytes_before = kv_bytes_before
            recipient_store = self._ssm_store
            recipient_unit = ssm_block_size
            recipient_bytes_before = ssm_bytes_before

        donor_bytes = batch_blocks * donor_unit
        recipient_bytes = (donor_bytes // recipient_unit) * recipient_unit
        bytes_wasted = donor_bytes - recipient_bytes

        monitor = self._pressure_monitor
        prev_state = self._current_pressure_state
        if monitor is not None and prev_state is not PoolPressureState.REBALANCING:
            monitor.record_transition(
                prev_state,
                PoolPressureState.REBALANCING,
                time.monotonic_ns(),
            )
        self._current_pressure_state = PoolPressureState.REBALANCING

        # Donor shrink leg.
        try:
            donor_store.resize_capacity(donor_bytes_before - donor_bytes)
        except (CapacityError, ValueError) as exc:
            self._current_pressure_state = prev_state
            elapsed = time.monotonic_ns() - start_ns
            return RebalanceOutcome(
                direction=direction,
                success=False,
                failure_reason=f"donor shrink rejected: {exc}",
                bytes_migrated=0,
                bytes_wasted_to_alignment=0,
                kv_pool_bytes_before=kv_bytes_before,
                kv_pool_bytes_after=kv_bytes_before,
                ssm_pool_bytes_before=ssm_bytes_before,
                ssm_pool_bytes_after=ssm_bytes_before,
                elapsed_ns=elapsed,
            )

        # Recipient grow leg.
        try:
            recipient_store.resize_capacity(recipient_bytes_before + recipient_bytes)
        except (CapacityError, ValueError) as exc:
            # Roll back the donor shrink. Restoring to donor_bytes_before is
            # always safe: it is a grow back to a previously-valid state on a
            # store whose backing tensor was pre-allocated to total_bytes.
            donor_store.resize_capacity(donor_bytes_before)
            self._current_pressure_state = prev_state
            elapsed = time.monotonic_ns() - start_ns
            return RebalanceOutcome(
                direction=direction,
                success=False,
                failure_reason=f"recipient grow rejected: {exc}",
                bytes_migrated=0,
                bytes_wasted_to_alignment=0,
                kv_pool_bytes_before=kv_bytes_before,
                kv_pool_bytes_after=kv_bytes_before,
                ssm_pool_bytes_before=ssm_bytes_before,
                ssm_pool_bytes_after=ssm_bytes_before,
                elapsed_ns=elapsed,
            )

        # Commit.
        elapsed = time.monotonic_ns() - start_ns
        self._rebalance_count += 1
        self._bytes_migrated_total += donor_bytes
        self._bytes_wasted_to_alignment_total += bytes_wasted
        self._time_spent_rebalancing_ns += elapsed

        kv_bytes_after = self._kv_store.num_total * kv_page_size
        ssm_bytes_after = self._ssm_store.num_total * ssm_block_size

        if monitor is not None:
            kv_free, ssm_free = self._free_ratios()
            new_state = monitor.compute_state(kv_free, ssm_free)
            monitor.record_transition(
                PoolPressureState.REBALANCING,
                new_state,
                time.monotonic_ns(),
            )
            self._current_pressure_state = new_state
        else:
            self._current_pressure_state = PoolPressureState.BALANCED

        return RebalanceOutcome(
            direction=direction,
            success=True,
            failure_reason=None,
            bytes_migrated=donor_bytes,
            bytes_wasted_to_alignment=bytes_wasted,
            kv_pool_bytes_before=kv_bytes_before,
            kv_pool_bytes_after=kv_bytes_after,
            ssm_pool_bytes_before=ssm_bytes_before,
            ssm_pool_bytes_after=ssm_bytes_after,
            elapsed_ns=elapsed,
        )
