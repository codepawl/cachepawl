# T001 — Establish vLLM baseline and AVMP integration path

Project: `.pawl/active/projects/project-main.md`
Sprint: `.pawl/active/sprints/sprint-001-vllm-integration.md`
Status: In Progress
Created: 2026-05-23
Updated: 2026-05-23
Completed: N/A
TTL: 30 days after completion or cancellation
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Objective

Set up the pinned vLLM environment, capture the vanilla hybrid-cache baseline, and begin the AVMP integration path described in `research/avmp/v2/VLLM_INTEGRATION_ROADMAP.md`.

## Current Behavior

Cachepawl has Python AVMP prototypes, benchmark tooling, and v2 Triton correctness-oracle artifacts, but no committed `src/cachepawl/integrations/vllm/` implementation yet.

## Expected Behavior

The repo has reproducible vLLM baseline evidence and a clear AVMP shim implementation path, with progress recorded in `.pawl/` and research artifacts updated as results land.

## Fix Strategy

- Follow `research/avmp/v2/VLLM_DEV_SETUP.md` for the local vLLM environment.
- Pin `vllm==0.21.0`.
- Try Zamba2-2.7B-instruct first, then Falcon-H1-1.5B-Instruct if the documented swap trigger fires.
- Record vanilla baseline results before changing vLLM integration behavior.
- Implement Path C unless constructor/private-state coupling forces the Path A fork fallback.

## Progress Notes

- 2026-05-23: Added the smallest product-code step for this task: an import-safe
  `cachepawl.integrations.vllm` skeleton. It exposes typed cache-plan records and
  availability helpers without importing or depending on vLLM.
- 2026-05-23: Updated README status wording so it no longer says nothing is
  implemented.
- 2026-05-23: Added a vLLM-free baseline measurement spine under
  `cachepawl.bench`: JSONL result schema, CPU-safe environment capture,
  deterministic synthetic workloads, RTX 3060 12GB config, and a planner probe
  CLI. This supports baseline planning records before runtime vLLM serving work.
- 2026-05-23: Added first planner-comparison evidence path: a clearly labeled
  vLLM-style padded modeling baseline versus Cachepawl AVMP planning on the same
  deterministic synthetic workloads, with JSONL records and Markdown/CSV
  summaries.
- 2026-05-23: Corrected benchmark memory-efficiency metric names before
  committing planner artifacts: `overestimation_ratio` is
  `estimated_bytes / useful_bytes`, and `wasted_fraction` is
  `(estimated_bytes - useful_bytes) / estimated_bytes`.
- 2026-05-23: Generated the first reproducible planner-comparison artifact pack
  under `benchmarks/results/rtx3060/planner-comparison/` for `short-heavy`,
  `long-heavy`, and `mixed` workloads with seed 1 and deterministic planner
  runtimes.

## Anti-Bypass Constraints

- Do not skip the vanilla baseline and claim AVMP improvement without paired evidence.
- Do not weaken tests or reduce workloads to hide allocator regressions.
- Do not use `TritonAVMPAllocator` as the production integration path for this sprint.
- Do not silently change paper claims; update the relevant research markdown when results change.

## Done When

- [ ] Vanilla vLLM serves the selected model or the fallback decision is recorded
- [ ] Baseline metrics are written under `research/avmp/v2/results/`
- [ ] AVMP shim code exists or Path A fallback is documented with evidence
- [x] Import-safe vLLM integration skeleton exists as the first shim step
- [x] Planner-only cache baseline measurement spine exists
- [x] Planner comparison emits vLLM-style padded versus Cachepawl AVMP evidence
- [x] Planner benchmark metrics use unambiguous ratio and fraction semantics
- [x] Reproducible RTX 3060 planner-comparison artifact pack exists
- [x] Verification commands and skipped checks are recorded
- [x] `.pawl/logs/changelog.md` summarizes the skeleton work

## Verification

Use the commands in `.pawl/context/REPO_COMMANDS.md`. For vLLM-specific checks, record exact commands, model, GPU, and any CUDA/WSL2 limitations.

2026-05-23 skeleton verification:

- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs view` — passed
- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration/vllm -q` — 7 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 137 files already formatted
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts` — passed, 135 source files
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 327 passed, 12 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements

Skipped checks are CUDA-dependent tests and the deferred v2.1 copy-region kernel test.

2026-05-23 baseline measurement spine verification:

- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs view` — passed
- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench -q` — 19 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 146 files already formatted
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py` — passed, 144 source files
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 346 passed, 12 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements
- `UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/run_cache_probe.py --workload short-heavy --backend avmp-static --seed 1 --num-requests 2 --gpu-total-bytes 12884901888` — emitted one JSONL record

2026-05-23 planner comparison verification:

- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs view` — passed
- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/compare_cache_planners.py --workload mixed --seed 1 --num-requests 8 --gpu-total-bytes 12884901888` — emitted Markdown summary
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench -q` — 27 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 151 files already formatted
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py` — passed, 149 source files
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 354 passed, 12 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements

2026-05-23 planner metric semantics verification:

- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs view` — passed
- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench -q` — 32 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/compare_cache_planners.py --workload mixed --seed 1 --num-requests 8 --gpu-total-bytes 12884901888` — emitted corrected Markdown summary with `overestimation_ratio` and `wasted_fraction`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 151 files already formatted
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py` — passed, 149 source files
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 359 passed, 12 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements

2026-05-23 planner artifact pack:

- Output directory: `benchmarks/results/rtx3060/planner-comparison/`
- Files: `README.md`, `environment.json`, `manifest.json`, `summary.md`, `short-heavy.jsonl`, `long-heavy.jsonl`, `mixed.jsonl`
- Generation command: `UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/create_planner_comparison_pack.py --output-dir benchmarks/results/rtx3060/planner-comparison --seed 1 --num-requests 128 --gpu-name "NVIDIA GeForce RTX 3060" --gpu-total-bytes 12884901888`
- The artifact records `vllm-style-padded` as a modeling baseline and `cachepawl-avmp` as planner-only evidence; no vLLM runtime serving or allocator replacement is included.
- `manifest.json` records artifact name, generation timestamp, fixed seed, request count, workloads, backends, target GPU profile, runtime-measurement setting, schema version, and generation command.
- Verification:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench -q` — 33 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 152 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py` — passed, 150 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 360 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed

## Regression Coverage

Added focused tests under `tests/integration/vllm/` for import safety, optional vLLM
availability probing, frozen/slots dataclass behavior, and Jamba reference cache-plan
translation. Runtime vLLM import/subclass tests remain pending until the vLLM venv is
available.

Added focused tests under `tests/bench/` for cache-probe schema round-trips and
validation, CPU-safe environment capture, deterministic synthetic workload output,
and CLI JSONL emission.

Added planner-comparison tests under `tests/bench/` for vLLM-style padded baseline
behavior, Cachepawl AVMP comparison behavior, schema round-trips, Markdown/CSV
summary rendering, and deterministic CLI JSONL output.

Added metric-semantics assertions for `overestimation_ratio` and
`wasted_fraction`, including the mixed workload example where the vLLM-style
padded baseline reports about 3.1266x overestimation and about 0.6802 wasted
fraction.

## Next Suggested Task

T002 — Run AVMP-enabled vLLM comparison once the baseline and shim are in place.
