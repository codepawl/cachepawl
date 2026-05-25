# Current Project Work

## Project

Cachepawl

## Current Sprint

`.pawl/active/sprints/sprint-001-vllm-integration.md`

## Current Task

`.pawl/active/tasks/t001-vllm-baseline-and-shim.md`

## Current Project

`.pawl/active/projects/project-main.md`

## Current Goal

Establish the vLLM baseline and implement the AVMP integration path described in `research/avmp/v2/VLLM_INTEGRATION_ROADMAP.md`.

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

Use the advisory diagnostic output to choose the smallest future mutation probe: planner-level hook, scheduler construction hook, or worker allocation hook.
