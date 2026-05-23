Review the current task.

Check:

- implementation matches the task
- no out-of-scope features were added
- full task details are in a task record file, not appended into an index
- root cause and fix strategy are recorded for bug-fix or implementation work
- verification proves the expected behavior without bypassing the issue
- tests/checks were not removed, skipped, weakened, or faked to make work pass
- validation, authorization, error handling, concurrency controls, and safety checks were not disabled to avoid the bug
- any hard-coded compatibility shim, broad mock, behavior deletion, or narrowed scope is explicitly required by the task or documented as a decision
- tests/typecheck/build status is recorded
- current sprint record and index are updated
- current task record and index are updated
- terminal records have `Completed`, `Archive After`, and `Archive Warning` set
- TTL warning notes or archive moves were applied where due

Then summarize:

- done
- not done
- verification and regression coverage
- TTL/archive status
- risks
- suggested next task
