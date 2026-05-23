# D001 — Use `.pawl/` as the project operating folder

Status: Accepted
Created: TODO
Updated: TODO
Completed: N/A
TTL: Keep while active
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Decision

This repository uses `.pawl/` as the source of truth for active planning, sprint tracking, task tracking, decisions, and project context.

## Reason

AI coding agents need stable repo-local context to avoid losing direction across sessions.

## Consequences

- Agents must read `.pawl/active/CURRENT.md` before work.
- Sprint and task updates must be written back to `.pawl/`.
- Archived files are historical context, not active instructions.

## Alternatives Considered

TODO

## Date

TODO
