# AGENTS.md

This repository uses `.pawl/` as the source of truth for project planning and progress.

Before making changes, coding agents must read:

1. `.pawl/active/CURRENT.md`
2. `.pawl/active/SPRINTS.md`
3. `.pawl/active/TASKS.md`
4. `.pawl/active/DECISIONS.md`
5. `.pawl/active/projects/INDEX.md`
6. `.pawl/active/sprints/INDEX.md`
7. `.pawl/active/tasks/INDEX.md`
8. `.pawl/active/decisions/INDEX.md`
9. `.pawl/context/PRODUCT_SCOPE.md`
10. `.pawl/context/TECHNICAL_SCOPE.md`
11. `.pawl/context/REPO_COMMANDS.md`

Rules:

- Work only on the current sprint and current task unless explicitly instructed otherwise.
- Do not implement future-sprint features early.
- Keep changes small and reviewable.
- Keep index files short. Put full project, sprint, task, and decision details in separate record files under `.pawl/active/{projects,sprints,tasks,decisions}/`.
- Follow existing project conventions.
- Do not add dependencies unless the task requires it and the reason is documented.
- For bug fixes, reproduce or characterize the current behavior before changing code, then record the root cause and verification.
- Fix causes, not symptoms. Do not hard-code around failing cases, delete behavior, narrow scope, or add broad mocks unless the task explicitly calls for that approach.
- Do not remove, skip, weaken, or fake tests/checks to make work pass.
- Do not disable validation, authorization, error handling, concurrency controls, or safety checks to avoid a bug.
- If the correct fix requires a scope or architecture change, stop and document the required decision instead of silently routing around it.
- Apply TTL policy after meaningful work: completed/cancelled/superseded records default to `TTL: 30 days`; add a warning note 7 days before `Archive After`; move expired records to the matching `.pawl/archive/` folder.
- Prefer `pawlkit view`, `pawlkit check`, `pawlkit integrate`, `pawlkit merge`, `pawlkit archive --dry-run`, `pawlkit archive`, and `pawlkit new ...` for structure maintenance when the CLI is available.
- Update `.pawl/active/CURRENT.md`, relevant active indexes, the relevant record file, and `.pawl/logs/changelog.md` after meaningful work.
- If scope or architecture changes, create or update a decision record under `.pawl/active/decisions/`.
- After work, summarize:
  - files changed
  - commands run
  - tests/typecheck/lint/build status
  - sprint/task status
  - remaining risks
