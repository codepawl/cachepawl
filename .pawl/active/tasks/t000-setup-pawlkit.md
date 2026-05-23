# T000 — Set up repo-local project operating files

Project: `.pawl/active/projects/project-main.md`
Sprint: `.pawl/active/sprints/sprint-000-pawlkit-setup.md`
Status: Completed
Created: 2026-05-23
Updated: 2026-05-23
Completed: 2026-05-23
TTL: 30 days after completion or cancellation
Archive After: 2026-06-22
Archive Warning: 2026-06-15
Archive Reason: Setup task completed

## Objective

Create the planning, context, and instruction files that future coding agents must use.

## Expected Files

- `.pawl/README.md`, `.pawl/active/CURRENT.md`, and short compatibility indexes
- `.pawl/active/{projects,sprints,tasks,decisions}/INDEX.md`
- Seed project, sprint, task, and decision record files
- `.pawl/context/*.md`, `.pawl/templates/*.md`, `.pawl/archive/*/README.md`, `.pawl/logs/changelog.md`
- `.agents/`, `.claude/`, `.codex/`, and root `AGENTS.md`

## Reproduction / Current Behavior

Fresh repositories do not yet have PawlKit operating files, so agents have no durable project/task source of truth.

## Expected Behavior

After setup, agents can read `.pawl/active/CURRENT.md`, find the current project/sprint/task files, follow repo constraints, and update progress consistently.

## Root Cause

The target repository has not been initialized with PawlKit.

## Fix Strategy

Install the default PawlKit scaffold without changing product code.

## Anti-Bypass Constraints

- Do not replace setup tracking with ad hoc notes outside `.pawl/`.
- Do not skip required operating files to make setup appear complete.
- Do not change product code while completing setup.

## Done When

- [x] All expected writable files exist
- [x] `AGENTS.md` points agents to `.pawl/`
- [x] Current project, sprint, and task are clear
- [x] No product code is changed
- [x] Changelog records the setup

## Verification

Confirmed `.pawl/`, `AGENTS.md`, and Claude integration files exist. `.agents/` and `.codex/` are read-only mounts in this environment and could not accept the optional adapter files.

Commands run:

- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check`
- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs view`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .`
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv build`

## Regression Coverage

The installed `AGENTS.md` and `.agents/workflow.md` require agents to keep using `.pawl/` for future task tracking.

## Next Suggested Task

T001 — Establish vLLM baseline and AVMP integration path.
