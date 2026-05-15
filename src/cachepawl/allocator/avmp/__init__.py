"""Asymmetric Virtual Memory Paging (AVMP) prototype.

See ``docs/designs/0001-asymmetric-virtual-memory-paging.md`` for the
full design rationale. This subpackage holds the primitives only:
handle dataclass, two physical backing stores, and the virtual page
table. The full ``Allocator`` implementation lands in a follow-up PR.
"""
