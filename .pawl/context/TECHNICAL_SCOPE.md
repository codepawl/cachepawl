# Technical Scope

## Stack

Python 3.10+, uv, pytest, hypothesis, ruff, mypy, torch, Triton, NumPy, matplotlib/Pillow for plots, and LaTeX research artifacts under `research/avmp/`.

## Package Manager / Build Tool

uv with `pyproject.toml` and `uv.lock`; hatchling builds the Python package.

## Test Command

`UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`

## Lint Command

`UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .`

## Typecheck Command

`UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts`

## Build Command

`UV_CACHE_DIR=/tmp/uv-cache uv build`

## Architecture Notes

- `src/cachepawl/allocator/avmp/` contains the AVMP allocator implementation, dynamic rebalancing state, physical/page-table helpers, and Triton-backed variant.
- `src/cachepawl/benchmarks/` contains workload, runner, aggregation, reporting, and plotting code used by committed benchmark artifacts.
- `research/avmp/v2/` contains the controlling vLLM integration audit, roadmap, development setup, and v2 Triton closeout documents.
- Public base APIs under `src/cachepawl/cache/`, `allocator/pool.py`, and `utils/device.py` still include intentional `NotImplementedError` surfaces.

## Constraints

- Keep changes small and reviewable
- Avoid unrelated refactors
- Avoid adding dependencies unless justified
- Prefer existing project conventions
- Keep `.pawl/active/*/INDEX.md` files short; store details in separate record files
- Archive completed/cancelled/superseded records after their 30-day TTL, with a warning 7 days before archive
- Fix root causes instead of masking symptoms
- Preserve public behavior unless a task explicitly changes it
- Do not remove, skip, weaken, or fake tests/checks to make work pass
- Do not hard-code around failing cases unless explicitly required
- Treat disabled validation, disabled error handling, broad mocks, deleted tests, and narrowed behavior as scope changes that must be documented
- Record skipped checks with the exact reason

## Known Risks

- CUDA-dependent tests skip in CPU-only environments.
- vLLM integration may require a fork fallback if Path C cannot cleanly subclass vLLM internals.
- README status text is partially stale relative to implemented allocator/benchmark code.
- `.agents/` and `.codex/` are read-only mounts in this workspace, so their optional PawlKit adapter files cannot be written here.
