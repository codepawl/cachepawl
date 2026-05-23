# Sprint 0 — PawlKit setup

Status: Completed
Created: 2026-05-23
Updated: 2026-05-23
Completed: 2026-05-23
TTL: 30 days after completion or cancellation
Archive After: 2026-06-22
Archive Warning: 2026-06-15
Archive Reason: Setup sprint completed

## Goal

Create repo-local operating files used by coding agents.

## Tasks

- [x] `.pawl/active/tasks/t000-setup-pawlkit.md`

## Definition of Done

- [x] Project operating files exist
- [x] Agents know what to read before work
- [x] Active project, sprint, task, and decision indexes are documented
- [x] Product and technical scope have initial notes
- [x] No product code was implemented

## Files Expected

- `.pawl/`
- `.agents/`
- `.claude/`
- `.codex/`
- `AGENTS.md`

## Non-Goals

- Do not implement product code.
- Do not add dependencies.

## Risks

- Large tracker files can become hard to scan if agents append full records instead of creating separate files.

## Notes

Sprint 0 installed the PawlKit tracker and initialized cachepawl-specific context. The environment exposes `.agents/` and `.codex/` as read-only mounts, so the root `AGENTS.md`, `.pawl/`, and Claude integration were installed; generic/Codex adapter files could not be written in this sandbox.
