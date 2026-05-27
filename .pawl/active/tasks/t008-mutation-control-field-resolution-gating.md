# T008 — Mutation-control field resolution and gating

Project: `.pawl/active/projects/project-main.md`
Sprint: `.pawl/active/sprints/sprint-006-mutation-control-field-gating.md`
Status: Completed
Created: 2026-05-26
Updated: 2026-05-27
Completed: 2026-05-27
TTL: 30 days after completion or cancellation
Archive After: 2026-06-26
Archive Warning: 2026-06-19
Archive Reason: Completed control-field go/no-go gate

## Objective

Inspect and classify the five mutation-required control fields identified by
T007, then decide whether T009 can be a controlled substitution experiment or
whether Cachepawl should remain advisory-only for this cycle.

## Starting Evidence

T007 completed mutation-readiness checks using:

- `research/avmp/v2/results/vllm-planner-stage-observation/translated_planner_stage_config.json`
- `research/avmp/v2/results/vllm-planner-stage-advisory-diff/diff_report.json`
- `research/avmp/v2/results/vllm-planner-stage-advisory-diff/group_level_diff.json`

All structural invariants passed:

- planner schema
- `num_blocks`
- cache group count
- cache tensor count
- layer coverage
- dtype/state dtype
- block/page size
- Mamba state shape
- attention/Mamba group mapping
- estimated/useful byte consistency

The readiness classification remains `advisory_only_recommended` because five
mutation-required fields are unresolved.

## Fields To Classify

For each field, classify it as exactly one of:

- `resolved_by_existing_evidence`
- `resolvable_by_read_only_observation`
- `requires_default_off_mutation_probe`
- `hard_blocker_for_current_cycle`

Fields:

1. stable scheduler or planner construction hook
2. allocator or KVCacheManager replacement control point
3. worker tensor allocation layout control point
4. runtime request-to-block assignment control
5. Mamba state-index and attention view rewrite contract

## Expected Output

Produce a control-field resolution report or decision artifact that records:

- classification for each field
- evidence used
- remaining risk
- whether T009 may be a controlled substitution experiment
- whether the project should stay advisory-only for this cycle
- rollback or stop conditions if a later default-off probe is allowed

Use `.pawl/active/decisions/d010-control-field-go-no-go-for-substitution.md`
as the go/no-go decision record template if the evidence is ready.

## Result

Created `research/avmp/v2/PATH_C_CONTROL_FIELD_GATE.md`.

Field classifications:

- stable scheduler or planner construction hook:
  `hard_blocker_for_current_cycle`
- allocator or KVCacheManager replacement control point:
  `hard_blocker_for_current_cycle`
- worker tensor allocation layout control point:
  `resolvable_by_read_only_observation`
- runtime request-to-block assignment control:
  `resolvable_by_read_only_observation`
- Mamba state-index and attention view rewrite contract:
  `resolvable_by_read_only_observation`

Final go/no-go classification: `stay_advisory_only_this_cycle`.

T009 should not be a controlled substitution experiment. If opened, it should
be a bounded read-only runtime contract observation task.

## Anti-Bypass Constraints

- Do not implement substitution.
- Do not modify vLLM source.
- Do not monkeypatch vLLM.
- Do not replace allocators.
- Do not alter scheduler behavior.
- Do not alter worker tensor layout.
- Do not add Triton kernels, copy kernels, LSDR, serving changes, or quality
  evaluation.

## Done When

- [x] Each of the five control fields has a recorded classification
- [x] The evidence and remaining risk for each field are documented
- [x] The T009 go/no-go result is documented
- [x] PawlKit validation is recorded

## Progress Notes

- 2026-05-26: Opened T008 as a gating task. It is explicitly
  pre-substitution and does not authorize mutation.
- 2026-05-27: Completed the control-field gate. D010 now keeps Cachepawl
  advisory-only for this cycle and recommends a read-only runtime contract
  observation as the smallest next task.

## Verification Log

2026-05-27 tracking setup:

- `npx @codepawl/pawlkit@0.3.0 view` — passed after approved npm access;
  initial sandboxed attempt failed with registry DNS `EAI_AGAIN`
- `npx @codepawl/pawlkit@0.3.0 check` — passed with 0 warnings after approved
  npm access; initial sandboxed attempt failed with registry DNS `EAI_AGAIN`

2026-05-27 control-field gate:

- `git diff --check` — passed
- `npx @codepawl/pawlkit@0.3.0 view` — passed after approved npm access;
  initial sandboxed attempt failed with registry DNS `EAI_AGAIN`
- `npx @codepawl/pawlkit@0.3.0 check` — passed with 0 warnings after approved
  npm access; initial sandboxed attempt failed with registry DNS `EAI_AGAIN`
