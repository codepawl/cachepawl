# Triton Audit: AVMP allocator GPU surface

Phase 1 audit of the current Python AVMP allocator, scoped to the question
"where do GPU operations happen today, and where do new Triton kernels need
to land to faithfully realize the simulation on real hardware?"

All file paths are relative to the repo root; line numbers cite the state of
`main` at the time of audit (commit `009bb2b`, branch `feat/v2-triton-scaffold`
forked from there).

## 1. `VirtualPageTable` API surface

Defined in `src/cachepawl/allocator/avmp/page_table.py:41-134`. Public methods:

| Method | Signature | Lines |
|---|---|---|
| `mint` | `(kind, virtual_offset, size_bytes, request_id, layer_idx, physical_offset) -> VirtualHandle` | 62-101 |
| `resolve` | `(handle_id) -> tuple[VirtualHandle, int]` | 103-116 |
| `remove` | `(handle_id) -> tuple[VirtualHandle, int]` | 118-133 |

State:

- Two `dict[int, _Entry]` partitioned by `HandleKind` (KV_PAGE vs SSM_BLOCK)
  so that O(1) `num_*_handles_live` counts do not need a full scan.
- `_next_handle_id: int` monotonic counter; **handle ids are never reused**
  within a single page table lifetime, so a stale `free(handle_id)` is
  rejected by `remove` (KeyError) rather than aliasing a fresh allocation.
- `_Entry` is a frozen dataclass holding `(VirtualHandle, physical_offset)`.

Invariant: every entry in either dict corresponds to an outstanding
`store.allocate_one()` call on the matching `KVPagesStore` or
`SSMBlocksStore`. The page table never owns tensor data.

## 2. `KVPagesStore` and `SSMBlocksStore`

Both defined in `src/cachepawl/allocator/avmp/physical.py:45-256`. Each
composes a `BackingStore` plus a `PageTable` from
`src/cachepawl/allocator/baselines/common.py`.

Page / block sizing:

- KV page size: `2 * num_kv_heads * head_dim * dtype_bytes * attention_page_tokens`
  (`physical.py:71`).
- SSM block size: `d_inner * d_state * dtype_bytes` (`physical.py:179`).

Tensor data:

- `BackingStore._buffer = torch.empty(total_bytes, dtype=torch.uint8, device=device)`
  (`baselines/common.py:76`). One per store; allocated once at
  construction time, never re-allocated. `resize_capacity` (RFC 0002 §4.4)
  changes only the `PageTable.set_num_pages_total()` counter, not the
  underlying tensor.

Public methods on each store: `allocate_one() -> int` (returns physical
offset), `free_one(offset) -> None`, `resize_capacity(new_capacity_bytes)
-> ResizeResult`, plus `page_size_bytes` / `block_size_bytes`, `num_total`,
`num_used`, `num_free` properties.

## 3. `allocate()` code path

`src/cachepawl/allocator/avmp/pool.py:181-325`. Sequence on a successful
allocation:

1. `AsymmetricVirtualPool.allocate(num_blocks, *, dtype_bytes)` (181-192)
   delegates kind selection to `_kind_from_context()` (186).
2. `_allocate_into(kind, num_blocks)` (308-325) calls
   `_try_bulk_allocate_with_eviction` and, on success, calls
   `VirtualPageTable.mint()` once per allocated block (314-322).
3. `_try_bulk_allocate_with_eviction(kind, num_blocks)` (327-354) calls
   `_bulk_allocate()` (367-368) which calls `store.allocate_one()` per
   block.
4. `_observe_pressure_state()` (191) updates monitor counters.

**GPU operations performed in this path: none.** No `torch.zeros`, no
`torch.empty`, no `.cuda()`, no `.to(device)`, no kernel launch. Every
`allocate_one()` returns an integer offset into the pre-allocated
`BackingStore._buffer`; the contents at that offset are whatever was left
from a prior free (uninitialized in the Python prototype).

Implication for the Triton path: a real hardware realization must
zero-fill the freshly allocated region before returning it (a real LLM
inference engine would otherwise read garbage). The `zero_page_kernel`
slot in the roadmap exists for exactly that reason.

## 4. `migrate_capacity()` / dynamic rebalancing path

State machine: `PoolPressureState` enum lives in `PoolPressureMonitor`
(`pool.py:165`, defined in `src/cachepawl/allocator/avmp/state.py`).
Values: `BALANCED=0`, `KV_PRESSURED=1`, `SSM_PRESSURED=2`,
`REBALANCING=3`.

`_apply_rebalance` (`pool.py:518-633`):

1. Compute donor/recipient pools from `RebalanceDirection`.
2. Call `donor.resize_capacity(donor.num_total * unit - batch_blocks * unit)`
   (shrink). On `CapacityError` (live pages block the shrink), return
   `RebalanceOutcome(success=False, failure_reason=...)` at 561-575.
3. Call `recipient.resize_capacity(recipient.num_total * unit + bytes)`
   (grow). On failure, **roll back** the donor shrink (584) and re-raise.
4. Accumulate `_bytes_migrated_total`, `_bytes_wasted_to_alignment_total`,
   `_time_spent_rebalancing_ns`.

No data copy. Migration is purely a counter adjustment on both
`PageTable.set_num_pages_total()` instances.

Implication for the Triton path: if the future cuMemMap-backed implementation
needs to physically remap pages (e.g., when an outstanding handle's physical
offset would now exceed the donor pool's new active capacity), a
`copy_region_kernel` is needed. If the realization keeps the same
"shrink only the active counter, leave live offsets untouched" semantics,
the kernel is unnecessary. This is open in the roadmap.

## 5. `CapacityError` flow

Raised in three places, all inside `src/cachepawl/allocator/baselines/common.py`:

- Class definition: line 33.
- `PageTable.alloc()` line 140 when `n > free_pages`.
- `PageTable.set_num_pages_total()` line 192 on shrink when a live page
  has offset >= the proposed new total.

Caught twice inside `_try_bulk_allocate_with_eviction` (`pool.py:330-354`):

1. **First catch (330)**: call `_evict_one()` and retry `_bulk_allocate`.
2. **Second catch (334)**: call
   `_maybe_auto_rebalance(forced=True, pressured_kind=kind)`.
   - If the forced rebalance succeeds (344), retry `_bulk_allocate` once
     more.
   - If it still fails, raise `torch.cuda.OutOfMemoryError` (347-354).

This is the only place migration fires today, per RFC 0002 §4.2: there is
no pre-emptive sampling hook. The observe-only updates in
`_observe_pressure_state` only affect counters, never trigger migration.

## 6. Existing CUDA operations

`grep -rn 'torch.cuda\|\.cuda()\|\.to(device)\|cuMemMap' src/cachepawl/allocator/avmp/`
returns:

- `pool.py:347, 351`: `torch.cuda.OutOfMemoryError` *raises only*, not GPU
  ops.

Plus one `torch.empty(total_bytes, dtype=torch.uint8, device=device)` per
`BackingStore` (`baselines/common.py:76`).

**Net surface: one tensor allocation per pool, zero kernel launches.**

Therefore every Triton kernel scheduled in
`research/avmp/v2/TRITON_ROADMAP.md` introduces **new behavior** to realize
the prototype on real hardware. None are ports of an existing GPU path.

## Appendix: `Allocator` ABC

`src/cachepawl/allocator/base.py:20-38` requires three methods:

```python
def allocate(self, num_blocks: int, *, dtype_bytes: int) -> list[int]: ...
def free(self, block_ids: Sequence[int]) -> None: ...
def stats(self) -> AllocatorStats: ...
```

`AsymmetricVirtualPool` additionally exposes two duck-typed methods that
the benchmark runner detects via `isinstance` against `_AllocatorContextProto`
and `_AllocatorStatsExporter`:

- `set_current_layer_kind(kind: LayerKind | None) -> None` (inherited from
  `AllocatorContext`).
- `set_current_request_id(request_id: str) -> None` (same).
- `get_allocator_stats() -> Mapping[str, float]` (`pool.py:239-299`).

A `TritonAVMPAllocator(AsymmetricVirtualPool)` subclass inherits all of
the above automatically; only the methods that gain kernel launches need
overrides.
