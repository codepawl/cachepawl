"""Post-sweep analysis utilities for the benchmarks comparison runner.

Each module here is intentionally small and single-purpose: ``stage_n``
parameter-sweep PRs add one analysis script each, and the running
collection of scripts stays grep-friendly. None of these modules touch
the allocator code path; they consume aggregated JSON files written by
``cachepawl.benchmarks.compare``.
"""

from cachepawl.benchmarks.analysis.lexicographic_rank import (
    VariantRanking,
    rank_variants,
)

__all__ = ["VariantRanking", "rank_variants"]
