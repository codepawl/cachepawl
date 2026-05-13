"""Smoke test for the cache submodule public surface."""

from __future__ import annotations

import abc

from cachepawl.cache import (
    HybridCacheCoordinator,
    KVCacheBlock,
    KVCacheManager,
    SSMStateBlock,
    StateCacheManager,
)
from cachepawl.quant import DType


def test_cache_managers_are_abstract() -> None:
    assert isinstance(KVCacheManager, abc.ABCMeta)
    assert isinstance(StateCacheManager, abc.ABCMeta)
    assert isinstance(HybridCacheCoordinator, abc.ABCMeta)


def test_kv_cache_block_dataclass() -> None:
    block = KVCacheBlock(block_id=0, layer_idx=3, num_tokens=16, dtype=DType.BF16)
    assert block.num_tokens == 16
    assert block.dtype is DType.BF16


def test_ssm_state_block_dataclass() -> None:
    block = SSMStateBlock(block_id=0, layer_idx=5, state_dim=128, dtype=DType.FP16)
    assert block.state_dim == 128
    assert block.dtype is DType.FP16
