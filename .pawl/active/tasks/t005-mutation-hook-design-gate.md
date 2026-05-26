# T005 — Mutation-hook design gate

Project: `.pawl/active/projects/project-main.md`
Sprint: `.pawl/active/sprints/sprint-003-path-c-mutation-hook-design.md`
Status: Completed
Created: 2026-05-26
Updated: 2026-05-26
Completed: 2026-05-26
TTL: 30 days after completion or cancellation
Archive After: 2026-06-25
Archive Warning: 2026-06-18
Archive Reason: Completed Path C mutation-hook design gate

## Objective

Compare bounded Path C control points for inserting or evaluating a
Cachepawl-proposed planner result after the successful T002 planner-stage
replay, without implementing mutation yet.

## Starting Evidence

T002 completed through the durable pinned vLLM environment at
`~/.cache/cachepawl/vllm-cachepawl-venv` with `vllm==0.21.0`. The replay reached
real planner-stage inputs and called
`get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory)` without
returning a Cachepawl plan to vLLM.

- `vllm_config` reached: true
- `kv_cache_specs` reached: true
- `available_memory` reached: true
- worker count: 1
- layer/spec count: 63
- spec types: `FullAttentionSpec=9`, `MambaSpec=54`
- available memory: `2915421184`
- planner output `num_blocks`: 329
- runtime scheduler `num_blocks`: 329
- `planner_matches_runtime_scheduler`: true
- `runtime_changed_during_replay`: false

## Candidate Control Points

T005 must compare:

1. Pre-call wrapper around `get_kv_cache_configs(...)`
2. Post-call advisory/diff only
3. Controlled return-value substitution in an isolated experiment
4. Scheduler/EngineCore hook

For each candidate, document:

- required control point
- correctness risk
- rollback strategy
- how to verify vLLM output unchanged in advisory mode
- how to verify changed behavior if substitution is later enabled
- required tests before mutation

## Output

Created `research/avmp/v2/PATH_C_MUTATION_HOOK_DESIGN_GATE.md` and accepted
D009. The selected next bounded experiment is a planner-stage post-call
advisory/diff artifact. Controlled return-value substitution remains deferred
until explicit default-off, opt-in, rollback, parity, and before/after artifact
safeguards are in place.

## Anti-Bypass Constraints

- Do not implement mutation in T005.
- Do not modify vLLM source.
- Do not monkeypatch vLLM yet.
- Do not replace allocators.
- Do not return any Cachepawl plan to vLLM.
- Do not alter scheduler behavior.
- Do not alter worker tensor layout.
- Do not add vLLM to the main Cachepawl dependencies.
- Do not add Triton kernels, copy kernels, LSDR, serving changes, or quality
  evaluation.

## Done When

- [x] The four candidate control points are compared
- [x] Required control point, correctness risk, rollback strategy, and
  verification approach are documented for each candidate
- [x] Required tests before mutation are listed
- [x] Advisory-mode unchanged-output verification is defined
- [x] Later substitution changed-behavior verification is defined
- [x] A recommendation or blocker is recorded
- [x] PawlKit validation is recorded

## Verification

Use the PawlKit commands in `.pawl/context/REPO_COMMANDS.md`. Product-code
tests are not required unless T005 unexpectedly changes product code.

## Progress Notes

- 2026-05-26: Opened T005 as a design-only Path C mutation-hook gate after the
  successful T002 planner-stage replay.
- 2026-05-26: Validated the tracker opening with PawlKit. Initial sandboxed
  `npx @codepawl/pawlkit@0.3.0 view` and `check` attempts failed with npm
  registry DNS `EAI_AGAIN`; approved network reruns passed, and `check`
  reported 0 warnings.
- 2026-05-26: Completed the design gate. Added
  `research/avmp/v2/PATH_C_MUTATION_HOOK_DESIGN_GATE.md` and D009 selecting a
  planner-stage post-call advisory/diff artifact as the next bounded
  experiment before any mutation.
- 2026-05-26: Final validation: `git diff --check` passed. Sandboxed PawlKit
  `view` and `check` failed with npm registry DNS `EAI_AGAIN`; approved
  network reruns passed, and `check` reported 0 warnings.
