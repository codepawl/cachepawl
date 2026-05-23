# `.pawl/archive/` — Historical Project Context

This folder holds expired projects, completed or cancelled sprints, completed or cancelled tasks, superseded decisions, abandoned ideas, and other historical notes.

## Rules

- **Archived files are not active instructions.** They are background only.
- **Agents may read archive for context**, but must base their work on `.pawl/active/` and `.pawl/context/`.
- **Active files override archive files.** If they conflict, `active/` wins.
- **Mirror active structure.** Move records into `projects/`, `sprints/`, `tasks/`, or `decisions/` to match their active location.
- **Leave a pointer.** After moving a record, leave a short archived pointer in the relevant active index.

## When to move something here

- A completed or cancelled project, sprint, or task has passed `Archive After`.
- A decision has been reversed, replaced, rejected, or is no longer relevant.
- A task or plan was abandoned and has passed `Archive After`.

## TTL policy

- Default TTL is `30 days` after completion, cancellation, supersession, or rejection.
- Default warning is `7 days before Archive After`.
- When the warning date is reached, add a warning note to the relevant active index.
- When `Archive After` is reached, move the record to the matching archive folder.
- Prefer `pawlkit archive --dry-run` before applying archive moves with `pawlkit archive`.

Keep filenames descriptive (e.g. `sprint-2-original-plan.md`, `d004-superseded-storage-choice.md`) so future readers can tell what they're looking at.
