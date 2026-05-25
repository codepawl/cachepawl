# Path C Mutation Probe Decision

Date: 2026-05-25

Status: choose a planner-level dry-run probe next; do not implement mutation in
this step.

## Inputs

- Decision gate: `research/avmp/v2/PATH_C_DECISION_GATE.md`
- Runtime diagnostic:
  `research/avmp/v2/results/vllm-runtime-cache-diagnostic/`
- Advisory result:
  - classification: `planner_advisory_available`
  - observed reserved bytes: `2910781440`
  - observed useful/recommended bytes: `1679258112`
  - advisory savings bytes: `1231523328`
  - overestimation ratio: `1.7333734577189286`
  - wasted fraction: `0.4230902777777778`

The diagnostic is advisory only. No vLLM source edits, monkeypatching,
allocator replacement, scheduler/manager/worker mutation, long-lived serving,
Triton kernels, copy kernels, LSDR, or quality evaluation have been performed.

## Candidate 1: Planner-Level Hook

### What It Would Observe Or Control

- Observe inputs and outputs around vLLM's planning functions before final
  `KVCacheConfig.kv_cache_tensors` sizes are committed.
- Candidate surfaces from the vLLM 0.21.0 audit:
  - `get_kv_cache_configs(...)`
  - `get_kv_cache_config_from_groups(...)`
  - `get_kv_cache_groups(...)`
  - page-size unification helpers
- A first probe should be dry-run only: call or wrap adjacent logic in an
  isolated helper to compare vLLM's planned tensor sizes with Cachepawl's
  advisory recommendation, then emit a proposed alternate plan without returning
  it to vLLM.

### Can Affect Runtime Behavior

Yes, eventually. This is the earliest candidate path that can test whether
Cachepawl recommendations can be inserted before vLLM finalizes cache tensor
sizes. A later mutation could change the `KVCacheConfig` produced by planning.

The first probe should not affect runtime behavior. It should only prove that
the required inputs are available at this point and that an alternate
Cachepawl-style plan can be computed next to the vanilla plan.

### Correctness Risk

Medium. The planner defines memory sizes consumed later by scheduler, manager,
and worker allocation paths. Incorrect changes could cause mismatched tensor
views, invalid block counts, or Mamba/attention shape incompatibilities.

The dry-run probe reduces risk because it does not return modified plans to
vLLM.

### Private API Risk

Medium-high. The candidate planning functions are vLLM internals, not a public
extension API. They are still the narrowest place to test the desired control
before final allocation layout.

### Testability On RTX 3060

Good for dry-run. The existing bounded `Zyphra/Zamba2-2.7B-instruct` load
already reaches runtime cache planning on the RTX 3060 with
`gpu_memory_utilization=0.7`, `max_model_len=4096`, and `max_num_seqs=1`.

### Expected Minimal Code Footprint

- Cachepawl-side helper that receives planner inputs or translated planning
  objects and emits a proposed alternate plan.
- A bounded script in the pinned vLLM environment that captures vanilla planner
  inputs/outputs and Cachepawl's proposed plan side by side.
- No vLLM source edits in the first probe.
- No production integration path yet.

### Rollback/Fallback Path

- If planner inputs cannot be reached without mutation, fall back to
  observer-only diagnostic CLI as the product step.
- If a future mutation requires changing vLLM internals, document a local fork
  or upstream extension request before implementation.

## Candidate 2: Scheduler Construction Hook

### What It Would Observe Or Control

- Observe or control `Scheduler.__init__` when it constructs `KVCacheManager`.
- Potential future control: inject a Cachepawl-aware manager or wrapper.

### Can Affect Runtime Behavior

Yes for scheduling and allocation behavior after construction. However, it is
probably too late to fix the specific overestimation shown by the diagnostic if
`KVCacheConfig.kv_cache_tensors` already carries padded tensor sizes.

### Correctness Risk

High. Scheduler construction connects admission, prefix caching, manager state,
coordinator behavior, and request block accounting. A wrong manager seam can
break scheduling invariants even if tensor allocation remains unchanged.

### Private API Risk

High. The audit found that vLLM 0.21.0 constructs `KVCacheManager` directly in
`Scheduler.__init__`; no stable factory seam was identified.

### Testability On RTX 3060

Moderate. A bounded load can instantiate the scheduler, but verifying behavior
requires request scheduling paths and likely generation smoke tests. This is
larger than the next minimal probe.

### Expected Minimal Code Footprint

Large relative to the planner dry-run. It would require subclass/wrapper design,
constructor compatibility checks, and a way to inject without monkeypatching or
source edits.

### Rollback/Fallback Path

- If no injection seam exists, defer to a local fork or upstream factory seam.
- Keep observer diagnostics as the non-mutating fallback.

## Candidate 3: Worker Allocation Hook

### What It Would Observe Or Control

- Observe or control `GPUModelRunner.initialize_kv_cache(...)`, raw cache tensor
  allocation, tensor reshaping, Mamba strided views, and attention cache views.

### Can Affect Runtime Behavior

Yes for actual tensor allocation layout. It is the closest path to memory
realization, but it is late relative to planning: scheduler and manager may
already assume the original `KVCacheConfig` sizes and block topology.

### Correctness Risk

Very high. The worker allocation path must preserve attention backend layout,
Mamba state indices, block tables, prefix-cache metadata, and view strides.
Changing this path without a planner-aligned config risks runtime shape or
indexing errors.

### Private API Risk

High. The path is internal and tightly coupled to vLLM worker and backend
implementation details.

### Testability On RTX 3060

Moderate to poor for a first probe. Bounded model load can reach allocation, but
meaningful validation needs generation and shape/index correctness checks. This
is larger and riskier than a planner dry-run.

### Expected Minimal Code Footprint

Largest of the three. It would require worker-path adapters plus correctness
checks for tensor views and Mamba/attention metadata.

### Rollback/Fallback Path

- Stop at observation if layout control cannot be validated safely.
- Prefer planner-level mutation first so worker allocation receives a coherent
  plan.

## Selected Next Probe

Choose the planner-level dry-run probe.

## Rationale

The decision preference is to choose the least invasive probe that can test
whether Cachepawl recommendations can be inserted before vLLM finalizes cache
planning. Among the three candidates, only the planner-level path is early
enough to affect `KVCacheConfig` tensor sizes while still allowing a first
probe to remain dry-run and non-mutating.

Scheduler construction is too late for the observed padded tensor-size issue
unless it also changes the already-created plan. Worker allocation is powerful
but too coupled to tensor views and backend metadata for the smallest next
probe.

## Probe Design

The next implementation should be a bounded dry-run planner probe:

1. Use the pinned `/tmp/vllm-cachepawl-venv` and `PYTHONPATH=src`.
2. Reuse the same target model and runtime bounds where feasible:
   `Zyphra/Zamba2-2.7B-instruct`, `max_model_len=4096`,
   `gpu_memory_utilization=0.7`, `max_num_seqs=1`.
3. Observe vanilla planner inputs and outputs adjacent to
   `get_kv_cache_configs(...)` or `get_kv_cache_config_from_groups(...)`.
4. Compute a Cachepawl proposed alternate plan from the same observed inputs.
5. Persist both the vanilla plan and the proposed plan as an artifact.
6. Do not return the proposed plan to vLLM.
7. Do not monkeypatch, modify vLLM source, replace allocators, or change runtime
   behavior.

## Success Criteria

- The probe records enough planner-stage inputs to reconstruct the vanilla
  runtime diagnostic metrics.
- The probe records a Cachepawl proposed alternate plan before finalized worker
  tensor allocation.
- The probe remains dry-run and read-only.
- If planner-stage inputs are not reachable without mutation, the fallback is an
  observer-only recommendation CLI rather than scheduler or worker mutation.

## Deferred Mutation Gate

Actual mutation remains blocked until a later decision proves:

- the planner-level control point is reachable;
- the alternate plan can preserve vLLM's scheduler, manager, worker, attention,
  and Mamba contracts;
- rollback to vanilla planning is explicit;
- bounded RTX 3060 smoke tests can compare vanilla versus proposed behavior.
