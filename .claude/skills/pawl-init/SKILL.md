---
name: pawl-init
description: Initialize or repair PawlKit project operating files in a software repository.
---

# Pawl Init Skill

Use this skill when the user asks to initialize PawlKit, set up repo planning files, repair missing project operating files, or prepare a repository for AI coding agents.

## Goal

Set up repo-local operating files that help coding agents track current work, sprints, tasks, decisions, product scope, technical scope, repo commands, and work logs.

## Rules

- Do not implement product code.
- Do not install dependencies.
- Do not overwrite existing `.pawl/` files without preserving user content.
- Prefer repair over replacement.
- Active files override archived files.
- Keep documents concise and operational.
- Keep indexes short; store full project, sprint, task, and decision details in separate record files.
- Add TTL fields to project, sprint, task, and decision records.

## Files To Create If Missing

- `.pawl/README.md`
- `.pawl/active/CURRENT.md`
- `.pawl/active/SPRINTS.md`
- `.pawl/active/TASKS.md`
- `.pawl/active/DECISIONS.md`
- `.pawl/active/projects/INDEX.md`
- `.pawl/active/projects/project-main.md`
- `.pawl/active/sprints/INDEX.md`
- `.pawl/active/sprints/sprint-000-pawlkit-setup.md`
- `.pawl/active/tasks/INDEX.md`
- `.pawl/active/tasks/t000-setup-pawlkit.md`
- `.pawl/active/decisions/INDEX.md`
- `.pawl/active/decisions/d001-use-pawl-folder.md`
- `.pawl/context/PRODUCT_SCOPE.md`
- `.pawl/context/TECHNICAL_SCOPE.md`
- `.pawl/context/DOMAIN_NOTES.md`
- `.pawl/context/REPO_COMMANDS.md`
- `.pawl/templates/project.md`
- `.pawl/templates/sprint.md`
- `.pawl/templates/task.md`
- `.pawl/templates/decision.md`
- `.pawl/archive/README.md`
- `.pawl/archive/projects/README.md`
- `.pawl/archive/sprints/README.md`
- `.pawl/archive/tasks/README.md`
- `.pawl/archive/decisions/README.md`
- `.pawl/logs/changelog.md`
- `.agents/instructions.md`
- `.agents/workflow.md`
- `AGENTS.md`

## Workflow

1. Inspect the repository.
2. Detect existing PawlKit files.
3. Create missing files.
4. Fill TODOs from existing README/config files when obvious.
5. Avoid speculative claims.
6. Summarize created, updated, skipped, and uncertain files.
