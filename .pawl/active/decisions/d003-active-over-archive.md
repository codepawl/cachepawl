# D003 — Active files override archive

Status: Accepted
Created: TODO
Updated: TODO
Completed: N/A
TTL: Keep while active
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Decision

Files under `.pawl/active/` are authoritative. Files under `.pawl/archive/` are historical.

## Reason

Archived plans may be outdated or superseded.

## Consequences

- Agents may read archive for context but must not treat it as current instruction.
- Active indexes may contain archived pointers, but active work must point to active record files.

## Alternatives Considered

TODO

## Date

TODO
