# Current Project Work

## Project

Cachepawl

## Current Sprint

`.pawl/active/sprints/sprint-002-planner-stage-observation.md`

## Current Task

`.pawl/active/tasks/t002-real-planner-stage-observation.md`

## Current Project

`.pawl/active/projects/project-main.md`

## Current Goal

Wait for restored host GPU/NVML access so T002 can rerun the bounded planner-stage observation around `get_kv_cache_configs(...)`; T003 is complete.

## Required Reading Before Work

- `.pawl/active/CURRENT.md`
- `.pawl/active/SPRINTS.md`
- `.pawl/active/TASKS.md`
- `.pawl/active/DECISIONS.md`
- `.pawl/active/projects/INDEX.md`
- `.pawl/active/sprints/INDEX.md`
- `.pawl/active/tasks/INDEX.md`
- `.pawl/active/decisions/INDEX.md`
- `.pawl/context/PRODUCT_SCOPE.md`
- `.pawl/context/TECHNICAL_SCOPE.md`
- `.pawl/context/REPO_COMMANDS.md`
- `AGENTS.md`

## Non-Goals Right Now

- Do not work outside the vLLM integration milestone unless explicitly instructed
- Do not change allocator behavior without task evidence and tests
- Do not revive the per-allocate Triton deployment path; that remains deferred to v2.1
- Do not weaken tests, lint, typecheck, or benchmark evidence to make progress appear complete

## Next Recommended Step

Restore host GPU/NVML visibility, then rerun the existing T002 planner-stage observation without changing vLLM, monkeypatching, replacing allocators, or continuing Path C mutation.
