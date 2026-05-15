"""Unit tests for ``cachepawl.allocator.avmp.physical``.

Both physical stores wrap shared baseline primitives. Tests pin the
properties the AVMP design relies on:

- Per-pool native page sizing, with no padding inflation between pools.
  This is the regression the vLLM unified pool exhibits and the central
  reason AVMP exists (RFC 0001 section 1).
- 128-byte alignment of every handed-out offset.
- ``num_used + num_free == num_total`` invariant across random ops.
- Capacity exhaustion raises :class:`CapacityError`.
- Cross-store independence: allocating in one does not deplete the
  other.
- FP4 is explicitly refused at construction.
"""

from __future__ import annotations

import dataclasses
import random

import pytest
import torch

from cachepawl.allocator.avmp.physical import KVPagesStore, SSMBlocksStore
from cachepawl.allocator.baselines.common import CapacityError, align_up
from cachepawl.allocator.baselines.padded_unified import PaddedUnifiedPool
from cachepawl.models.spec import HybridModelSpec
from cachepawl.quant.dtypes import DType, bytes_per_element

_KV_TOTAL_BYTES = 4 * 1024 * 1024  # 4 MiB, enough for many pages on jamba spec
_SSM_TOTAL_BYTES = 16 * 1024 * 1024  # 16 MiB, fits a handful of large SSM blocks


def _expected_kv_page_size(spec: HybridModelSpec, attention_page_tokens: int) -> int:
    elem = bytes_per_element(spec.dtype)
    prof = spec.attention_profile
    raw = max(1, int(2.0 * prof.num_kv_heads * prof.head_dim * elem * attention_page_tokens))
    return align_up(raw)


def _expected_ssm_block_size(spec: HybridModelSpec) -> int:
    elem = bytes_per_element(spec.dtype)
    prof = spec.ssm_profile
    raw = max(1, int(prof.d_inner * prof.d_state * elem))
    return align_up(raw)


def test_kv_page_size_matches_formula(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = KVPagesStore(
        model_spec=jamba_spec,
        attention_page_tokens=16,
        total_bytes=_KV_TOTAL_BYTES,
        device=cpu_device,
    )
    assert store.page_size_bytes == _expected_kv_page_size(jamba_spec, 16)
    assert store.page_size_bytes % 128 == 0


def test_ssm_block_size_matches_formula(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = SSMBlocksStore(
        model_spec=jamba_spec,
        total_bytes=_SSM_TOTAL_BYTES,
        device=cpu_device,
    )
    assert store.block_size_bytes == _expected_ssm_block_size(jamba_spec)
    assert store.block_size_bytes % 128 == 0


def test_allocate_one_returns_ascending_aligned_offsets(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = KVPagesStore(
        model_spec=jamba_spec,
        attention_page_tokens=16,
        total_bytes=_KV_TOTAL_BYTES,
        device=cpu_device,
    )
    page_size = store.page_size_bytes
    offsets = [store.allocate_one() for _ in range(5)]
    for offset in offsets:
        assert offset % page_size == 0
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets)


def test_occupancy_invariant_under_random_ops(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = KVPagesStore(
        model_spec=jamba_spec,
        attention_page_tokens=16,
        total_bytes=_KV_TOTAL_BYTES,
        device=cpu_device,
    )
    rng = random.Random(42)
    live: list[int] = []
    for _ in range(50):
        if live and rng.random() < 0.5:
            offset = live.pop(rng.randrange(len(live)))
            store.free_one(offset)
        elif store.num_free > 0:
            live.append(store.allocate_one())
        assert store.num_used + store.num_free == store.num_total
        assert store.num_used == len(live)


def test_free_one_then_allocate_one_reuses_offset(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = SSMBlocksStore(
        model_spec=jamba_spec,
        total_bytes=_SSM_TOTAL_BYTES,
        device=cpu_device,
    )
    first = store.allocate_one()
    second = store.allocate_one()
    store.free_one(first)
    third = store.allocate_one()
    assert third == first
    assert second != first


def test_allocate_one_raises_capacity_error_when_exhausted(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = KVPagesStore(
        model_spec=jamba_spec,
        attention_page_tokens=16,
        total_bytes=_KV_TOTAL_BYTES,
        device=cpu_device,
    )
    for _ in range(store.num_total):
        store.allocate_one()
    with pytest.raises(CapacityError):
        store.allocate_one()


def test_free_one_rejects_misaligned_offset(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = KVPagesStore(
        model_spec=jamba_spec,
        attention_page_tokens=16,
        total_bytes=_KV_TOTAL_BYTES,
        device=cpu_device,
    )
    store.allocate_one()
    with pytest.raises(ValueError, match="not a multiple of"):
        store.free_one(7)


def test_kv_and_ssm_stores_are_independent(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    kv = KVPagesStore(
        model_spec=jamba_spec,
        attention_page_tokens=16,
        total_bytes=_KV_TOTAL_BYTES,
        device=cpu_device,
    )
    ssm = SSMBlocksStore(
        model_spec=jamba_spec,
        total_bytes=_SSM_TOTAL_BYTES,
        device=cpu_device,
    )
    ssm_used_before = ssm.num_used
    kv.allocate_one()
    kv.allocate_one()
    assert ssm.num_used == ssm_used_before


def test_kv_page_smaller_than_padded_unified_page(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    """Per-pool sizing is the AVMP design point versus the vLLM unified pool.

    On a hybrid spec where the SSM block bytes exceed the KV page bytes
    (the common case for Mamba-Transformer hybrids), the unified pool
    inflates the KV page to the SSM size, while AVMP keeps the KV pool
    at its native page size.
    """

    kv = KVPagesStore(
        model_spec=jamba_spec,
        attention_page_tokens=16,
        total_bytes=_KV_TOTAL_BYTES,
        device=cpu_device,
    )
    unified = PaddedUnifiedPool(
        model_spec=jamba_spec,
        total_bytes=_KV_TOTAL_BYTES,
        device=cpu_device,
        attention_page_tokens=16,
    )
    unified_page_bytes = int(unified.get_allocator_stats()["page_size_bytes"])
    assert kv.page_size_bytes < unified_page_bytes


def test_fp4_dtype_is_rejected_with_pointer_to_rfc(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    fp4_spec = dataclasses.replace(jamba_spec, dtype=DType.FP4)
    with pytest.raises(NotImplementedError, match=r"section 3\.3"):
        KVPagesStore(
            model_spec=fp4_spec,
            attention_page_tokens=16,
            total_bytes=_KV_TOTAL_BYTES,
            device=cpu_device,
        )
    with pytest.raises(NotImplementedError, match=r"section 3\.3"):
        SSMBlocksStore(
            model_spec=fp4_spec,
            total_bytes=_SSM_TOTAL_BYTES,
            device=cpu_device,
        )
