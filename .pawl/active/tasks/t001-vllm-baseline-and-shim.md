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

## Regression Coverage

Added focused tests under `tests/integration/vllm/` for import safety, optional vLLM
availability probing, frozen/slots dataclass behavior, and Jamba reference cache-plan
translation. Runtime vLLM import/subclass tests remain pending until the vLLM venv is
available.

## Next Suggested Task

T002 — Run AVMP-enabled vLLM comparison once the baseline and shim are in place.
