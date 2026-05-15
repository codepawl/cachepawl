"""Virtual handle types for AVMP.

Reference design: ``docs/designs/0001-asymmetric-virtual-memory-paging.md``
sections 3.1 and 4.

Conceptual prior art: vTensor (`arXiv:2407.15309
<https://arxiv.org/abs/2407.15309>`_) demonstrates how a virtual handle
namespace decoupled from the physical backing slabs lets each pool keep
its native page size. CUDA's Virtual Memory Management API
(``cuMemCreate`` / ``cuMemMap`` / ``cuMemAddressReserve``) is the
eventual native target referenced in the RFC; this Python prototype
stays in PyTorch tensors only.

``VirtualHandle`` is caller-visible metadata. It is **not** the
``int`` block id returned by :meth:`Allocator.allocate`; the
``AsymmetricVirtualPool`` (landing in the follow-up PR) implements the
ABC's ``allocate(num_blocks, *, dtype_bytes) -> list[int]`` signature
and keys ``VirtualHandle`` records by those returned ids.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class HandleKind(enum.Enum):
    """Which physical pool a :class:`VirtualHandle` resolves into."""

    KV_PAGE = "kv_page"
    SSM_BLOCK = "ssm_block"


@dataclass(frozen=True, slots=True)
class VirtualHandle:
    """Opaque-to-callers record describing one AVMP allocation.

    Field semantics:

    - ``handle_id``: unique non-zero integer minted by the page table.
      Never reused within a single :class:`VirtualPageTable` lifetime
      (handle ids are not VRAM bytes; non-reuse keeps a use-after-free
      bug visible rather than silently colliding with a fresh
      allocation).
    - ``kind``: which pool the handle resolves into. Determined at
      mint time.
    - ``virtual_offset``: position inside the per-kind virtual address
      space. Distinct from the physical offset stored alongside the
      handle in the page table; how the pool assigns virtual offsets is
      left to the pool implementation in the follow-up PR.
    - ``size_bytes``: logical bytes the caller asked for (not the
      aligned page or block bytes, which may be larger).
    - ``request_id``: caller-supplied request identifier. Kept as a
      string for debug and log line clarity; the pool maps this to the
      integer key used by :class:`LRURequestTracker` when needed.
    - ``layer_idx``: which model layer the handle was minted for, so
      per-layer accounting works without a second pass over the table.
    """

    handle_id: int
    kind: HandleKind
    virtual_offset: int
    size_bytes: int
    request_id: str
    layer_idx: int
