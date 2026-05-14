"""Module entrypoint so ``python -m cachepawl.benchmarks.compare`` works."""

from cachepawl.benchmarks.compare.sweep import main

if __name__ == "__main__":
    raise SystemExit(main())
