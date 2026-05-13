"""Dummy cache workload benchmark.

Drives a concrete allocator with synthetic reservation traces to measure
fragmentation and bandwidth before any model is wired up. This file is a
stub. The TODO list below tracks what needs to land alongside the first
concrete allocator before this file is useful.

TODOs:
- Wire MemoryPool against a real backing tensor store.
- Generate adversarial sequence shape traces (short and long mixed).
- Measure reservation throughput in ops per second.
- Measure fragmentation ratio after each trace.
- Measure achievable bandwidth on a synthetic read-heavy workload.
- Emit a CSV summary that can be diffed across allocator revisions.
"""

from __future__ import annotations


def main() -> None:
    """Entry point for the dummy cache workload benchmark."""

    raise NotImplementedError(
        "dummy_cache_workload.main: implement once MemoryPool has a real backend."
    )


if __name__ == "__main__":
    main()
