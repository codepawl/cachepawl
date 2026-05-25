# Repo Commands

## Install

`uv sync --extra-index-url https://download.pytorch.org/whl/cpu`

## Build

`UV_CACHE_DIR=/tmp/uv-cache uv build`

## Test

`UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`

## Typecheck

`UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts`

## Lint

`UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .`

## Format

`UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .`

## PawlKit

`npx @codepawl/pawlkit@0.3.0 view`

`npx @codepawl/pawlkit@0.3.0 check`

## Run

No long-running app entrypoint is defined. Use benchmark and research scripts for targeted workflows.

## Notes

- Use `UV_CACHE_DIR=/tmp/uv-cache` in this sandbox because the default uv cache under the home directory is read-only.
- Use the pinned scoped PawlKit package when `pawlkit` is not on `PATH`; `npx pawlkit` is not the correct package.
- CI runs unit tests with `uv run pytest tests/unit -v` on Python 3.10 and 3.12.
- CUDA-dependent tests are expected to skip without a CUDA-capable GPU.
