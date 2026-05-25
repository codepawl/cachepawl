# D008 — Dogfood PawlKit Through Cachepawl Only When Friction Appears

Status: Accepted
Created: 2026-05-25
Updated: 2026-05-25
Completed: N/A
TTL: Keep while active
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Decision

Use normal Cachepawl tracker work as lightweight PawlKit dogfooding.

Record PawlKit issues only when they create real friction during normal
Cachepawl work, such as repository friction, validation issues, command
ambiguity, performance cost, or workflow bugs.

Do not proactively test PawlKit beyond normal `view`, `check`, and tracker
update usage. Do not block Cachepawl research tasks on PawlKit improvements
unless PawlKit validation itself is broken.

If a PawlKit issue is found, record:

- command used
- expected behavior
- actual behavior
- impact on Cachepawl
- workaround
- suggested PawlKit fix

If no PawlKit issue occurs, do nothing.

## Reason

Cachepawl is using PawlKit as its real project tracker, so normal tracker work
is the useful feedback source. Extra PawlKit testing would distract from the
active vLLM integration research and could create noise that is not tied to
Cachepawl's actual workflow.

## Consequences

- PawlKit feedback stays grounded in real Cachepawl workflow friction.
- Cachepawl's current vLLM integration work remains the primary work.
- Passing `view` and `check` without friction requires no extra PawlKit note.
- Broken PawlKit validation can block tracker updates until the validation
  issue is captured and worked around.

## Date

2026-05-25
