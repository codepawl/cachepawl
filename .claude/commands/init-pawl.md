Initialize or repair PawlKit in this repository.

Read:

- `AGENTS.md` if present
- `.pawl/README.md` if present
- `.pawl/active/CURRENT.md` if present
- `.pawl/active/projects/INDEX.md` if present
- `.pawl/active/sprints/INDEX.md` if present
- `.pawl/active/tasks/INDEX.md` if present
- `.pawl/active/decisions/INDEX.md` if present
- `.pawl/context/PRODUCT_SCOPE.md` if present
- repository README and package/config files

Task:

- If PawlKit files do not exist, create them using the default structure.
- If PawlKit files already exist, repair missing files only.
- Do not overwrite active files without showing a diff-style summary first.
- Keep indexes short and put full records in `.pawl/active/{projects,sprints,tasks,decisions}/`.
- Add TTL fields to new project, sprint, task, and decision records.
- Infer product and technical scope from the repo when possible.
- Leave TODOs when unsure.
- Do not implement product code.
