# `.pawl/` — Repo-Local Project Operating Folder

`.pawl/` is the source of truth for how this repository is currently being worked on. It exists so AI coding agents (and humans collaborating with them) can pick up work mid-stream without losing direction.

## What lives here

- **`active/`** — the authoritative, current state of the project:
  - `CURRENT.md` — the active dashboard: project, current sprint, current task, required reading.
  - `SPRINTS.md`, `TASKS.md`, `DECISIONS.md` — short compatibility indexes.
  - `projects/` — short project index plus one file per active project.
  - `sprints/` — short sprint index plus one file per active sprint.
  - `tasks/` — short task index plus one file per active task.
  - `decisions/` — short decision index plus one file per active decision.
- **`context/`** — durable project context:
  - `PRODUCT_SCOPE.md` — what the project is and is not.
  - `TECHNICAL_SCOPE.md` — stack, constraints, build/test/lint commands.
  - `DOMAIN_NOTES.md` — vocabulary, assumptions, non-obvious constraints.
  - `REPO_COMMANDS.md` — the commands agents should use.
- **`templates/`** — reusable templates for new projects, sprints, tasks, and decisions.
- **`archive/`** — historical records, mirrored by projects, sprints, tasks, and decisions.
- **`logs/`** — the running work log (`changelog.md`).

## Rules

1. **Agents must read `.pawl/active/CURRENT.md` before doing any work.** It points at everything else.
2. **Active files override archived files.** If an active record and an archived record disagree, the active file wins.
3. **Trackers must be updated after meaningful changes.** When a task moves, when a decision is made, when a sprint advances — write it back here. A stale `.pawl/` is worse than no `.pawl/`.
4. **Do not start future-sprint work early.** If it is not in the current sprint, it is not in scope.
5. **Finish with evidence.** For bug fixes and implementation tasks, record current behavior, expected behavior, root cause, fix strategy, verification, and regression coverage before marking work done.
6. **Keep files short.** Index files should contain pointers and status tables only; full details belong in one record file per project, sprint, task, or decision.
7. **Apply TTL.** Completed, cancelled, or superseded records default to `TTL: 30 days`; warn 7 days before archiving and move expired records to the matching archive folder.

## Maintenance Commands

Use the pinned package when `pawlkit` is not already on `PATH`:

- `npx @codepawl/pawlkit@0.3.0 view`
- `npx @codepawl/pawlkit@0.3.0 check`

The unscoped package name `pawlkit` is not the published npm package for this
repo's current tooling.

- `pawlkit check` — validate structure, index size, `.pawl` links, and required metadata.
- `pawlkit view` — print a read-only terminal dashboard of current records, statuses, and TTL labels.
- `pawlkit integrate` — create or repair Claude, Codex, generic agent files, and `AGENTS.md`.
- `pawlkit merge` — upgrade older PawlKit `.pawl/` folders into the split-file structure.
- `pawlkit archive --dry-run` — report TTL warnings and expired records without changing files.
- `pawlkit archive` — move expired terminal records to archive and leave index pointers.
- `pawlkit new task "Title"` — create a record from a template; also supports `project`, `sprint`, and `decision`.
