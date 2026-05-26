# Path C Mutation-Hook Design Gate

Date: 2026-05-26

Status: select planner-stage post-call advisory/diff as the next bounded
experiment; do not implement mutation yet.

## Inputs

- Planner-stage artifact:
  `research/avmp/v2/results/vllm-planner-stage-observation/`
- Function reached:
  `vllm.v1.core.kv_cache_utils.get_kv_cache_configs`
- Durable vLLM environment:
  `~/.cache/cachepawl/vllm-cachepawl-venv`
- vLLM version: `0.21.0`

T002 successfully replayed vanilla planner-stage config generation without
modifying vLLM, monkeypatching, replacing allocators, returning Cachepawl plans,
altering scheduler behavior, altering worker tensor layout, adding kernels,
serving changes, LSDR, or quality evaluation.

## T002 Evidence

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

This proves Cachepawl can observe the real planner boundary and translate the
vanilla output. It does not prove that substituting a different result is safe.

## Candidate Comparison

| Path | Where it attaches | Monkeypatching required | Changes vLLM behavior | Correctness risk | Private API risk | Rollback strategy | RTX 3060 testability | Required evidence before implementation | Minimal code footprint |
|---|---|---:|---:|---|---|---|---|---|---|
| Pre-call wrapper around `get_kv_cache_configs(...)` | Immediately before the planner function receives `vllm_config`, `kv_cache_specs`, and `available_memory` | Yes, unless vLLM exposes an injection seam or a local fork is used | Not in advisory mode; yes if it alters inputs or intercepts the call | Medium-high: input mutation can desynchronize planner assumptions before config construction | High: planner entrypoint is an internal function | Default off, remove wrapper, fall back to vanilla direct replay artifact | Good for bounded model-load replay on RTX 3060 | Stable hook seam, input immutability proof, vanilla parity artifact, proposed-plan artifact, failure artifact | Moderate: wrapper plus artifact writer |
| Post-call advisory/diff only | Immediately after vanilla `get_kv_cache_configs(...)` returns, before any Cachepawl result is returned to vLLM | No for replay-side artifact generation; yes only if implemented inline inside vLLM execution without an extension seam | No | Low: observes vanilla output and computes a sidecar recommendation | Medium: still depends on internal planner outputs, but does not alter them | Delete advisory hook/script output; vLLM remains vanilla | Excellent: T002 already proves the boundary on RTX 3060 | Deterministic before/after artifact, vanilla scheduler parity, advisory recommendation schema, unsupported-field diagnostics | Small: extend current planner-stage artifact path with sidecar diff |
| Controlled isolated return-value substitution | At the return value from `get_kv_cache_configs(...)`, replacing vanilla `KVCacheConfig` only in an explicit experiment | Yes unless done in a local fork or sanctioned extension seam | Yes, when enabled | High: substituted config must satisfy scheduler, manager, worker, attention, and Mamba contracts | High: no public substitution API identified | Default-off flag, explicit opt-in, immediate fallback to vanilla on validation failure, separate artifact directory | Moderate: bounded RTX 3060 load can prove initialization behavior; broader behavior needs later tests | Cachepawl config contract proof, vanilla parity baseline, changed-behavior artifact, rollback artifact, tensor-layout compatibility checks | Medium-high: guarded substitution path plus validators |
| Scheduler/EngineCore hook | Around `EngineCore._initialize_kv_caches(...)` or scheduler construction after planner output exists | Likely yes without vLLM source/fork support | Yes if it swaps scheduler config or manager state | High: scheduler, manager, prefix-cache, and block accounting invariants are coupled | High: construction paths are private internals | Disable hook, use vanilla scheduler config, fall back to post-call advisory | Moderate: load reaches scheduler, but meaningful validation quickly expands beyond design gate scope | Stable construction seam, manager/scheduler invariant checks, request-admission parity, worker allocation compatibility | Large: hook, compatibility shims, and broader tests |

## Recommendation

Choose post-call advisory/diff only as the next bounded experiment.

This path is the safest next step because it uses the already-proven T002
planner boundary, preserves vanilla vLLM behavior, requires no monkeypatch in
normal artifact/replay mode, and can produce the missing evidence for a later
substitution decision. It should compute a Cachepawl proposed planner result
beside the vanilla planner result and persist a structured diff, but it must not
return the Cachepawl proposal to vLLM.

Controlled return-value substitution remains the first plausible mutation
experiment, but only after the advisory/diff artifact proves contract
compatibility and defines validation gates. Pre-call wrapping and
Scheduler/EngineCore hooks are higher-risk paths and should stay deferred unless
post-call diff evidence shows substitution cannot validate the needed contract.

## Required Safeguards For Any Future Mutation

Any future mutation task must require:

- default-off feature flag
- explicit opt-in
- no mutation in normal CLI/advisory mode
- structured before/after artifact
- rollback path to vanilla planner output
- parity check against vanilla vLLM
- refusal to proceed when required fields or invariants are missing
- no vLLM source edits unless a separate decision explicitly selects a fork
- no scheduler or worker tensor layout mutation without a later dedicated gate

## Advisory-Mode Verification

The next bounded experiment should prove unchanged vanilla behavior by recording
all of the following in one artifact:

- vanilla planner output translated through Cachepawl
- runtime scheduler config translated after initialization
- `planner_matches_runtime_scheduler`
- `runtime_changed_during_replay`
- vanilla prompt/generation behavior unchanged only if a later task explicitly
  includes generation smoke; this design gate does not require quality
  evaluation
- Cachepawl proposed plan and diff stored only as sidecar data
- object-access flags showing no returned Cachepawl plan, no monkeypatching, no
  allocator replacement, no scheduler mutation, and no worker layout mutation

## Later Substitution Verification

If a later task enables controlled substitution, it must produce a separate
before/after artifact that proves:

- vanilla baseline and substituted run use the same model, vLLM version, runtime
  bounds, and durable environment
- feature flag is default off and explicit opt-in is recorded
- vanilla planner output, Cachepawl proposed output, and substituted output are
  all persisted
- the substituted config passes structural checks before it is returned
- changed behavior is intentional and limited to the selected cache-planning
  fields
- rollback to vanilla planner output is exercised or documented
- failures produce structured blockers instead of falling through to partially
  mutated runtime state

## Required Tests Before Mutation

Before any mutation implementation, add tests or artifacts for:

- deterministic post-call advisory/diff generation from the T002 translated
  planner output
- unsupported/missing-field reporting for incomplete planner metadata
- structural compatibility checks for any proposed `KVCacheConfig`
- default-off feature flag behavior
- explicit opt-in requirement
- rollback behavior when validation fails
- no-vLLM import safety for any Cachepawl-side helpers

## Decision

Open the next implementation task as a planner-stage post-call advisory/diff
artifact. Do not implement mutation yet.
