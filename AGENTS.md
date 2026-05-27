# AGENTS.md

Before making changes, coding agents should read:

1. `README.md`
2. `CLAUDE.md`
3. Relevant design or research documents for the area being changed

Rules:

- Keep changes small and reviewable.
- Follow existing project conventions.
- Do not add dependencies unless the task requires it and the reason is clear.
- For bug fixes, reproduce or characterize the current behavior before changing code, then record the root cause and verification.
- Fix causes, not symptoms.
- Do not remove, skip, weaken, or fake tests/checks to make work pass.
- Do not disable validation, authorization, error handling, concurrency controls, or safety checks to avoid a bug.
- If a correct fix requires a scope or architecture change, document the tradeoff instead of silently routing around it.

After work, summarize:

- files changed
- commands run
- tests/typecheck/lint/build status
- remaining risks
