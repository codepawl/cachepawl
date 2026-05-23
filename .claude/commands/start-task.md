Read the current task from `.pawl/active/CURRENT.md`, `.pawl/active/tasks/INDEX.md`, and the current task record file.

Implement only that task.

Before coding:

- identify files likely to change
- confirm the current project, sprint file, and task file
- confirm the task belongs to the current sprint
- avoid future-sprint work
- for bug fixes, reproduce or characterize current behavior
- identify expected behavior, likely root cause, and anti-bypass constraints

During coding:

- fix causes instead of masking symptoms
- preserve public behavior unless the task explicitly changes it
- do not remove, skip, weaken, or fake tests/checks to make work pass
- do not hard-code around failing cases, delete behavior, narrow scope, add broad mocks, or disable validation/error handling unless the task explicitly requires it

After coding:

- run relevant checks
- record root cause, fix strategy, verification, regression coverage, and remaining risks
- update `.pawl/active/CURRENT.md`
- update the relevant short indexes under `.pawl/active/{projects,sprints,tasks,decisions}/`
- update the current project, sprint, and task record files
- apply TTL warnings or archive moves when records are due
- update `.pawl/logs/changelog.md`
