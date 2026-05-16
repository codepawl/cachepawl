"""resize_capacity stubs raise NotImplementedError in v2 sub-PR 1.

DELETE this file in sub-PR 2 when the resize_capacity implementation lands;
the migration mechanics tests there will exercise the real behavior.
"""

from __future__ import annotations

import pytest
import torch

from cachepawl.allocator.avmp.physical import KVPagesStore, SSMBlocksStore
from cachepawl.models.spec import HybridModelSpec


def test_kv_pages_store_resize_capacity_stub(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = KVPagesStore(
        model_spec=jamba_spec,
        attention_page_tokens=16,
        total_bytes=4 * 1024 * 1024,
        device=cpu_device,
    )
    with pytest.raises(NotImplementedError, match="migration mechanics land in v2 sub-PR 2"):
        store.resize_capacity(1 * 1024 * 1024)


def test_ssm_blocks_store_resize_capacity_stub(
    jamba_spec: HybridModelSpec,
    cpu_device: torch.device,
) -> None:
    store = SSMBlocksStore(
        model_spec=jamba_spec,
        total_bytes=4 * 1024 * 1024,
        device=cpu_device,
    )
    with pytest.raises(NotImplementedError, match="migration mechanics land in v2 sub-PR 2"):
        store.resize_capacity(1 * 1024 * 1024)
