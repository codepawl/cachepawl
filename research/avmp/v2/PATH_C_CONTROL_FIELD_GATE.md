# Path C Control-Field Gate

Date: 2026-05-27

Status: stay advisory-only this cycle; do not open a controlled substitution
experiment yet.

## Scope

This gate resolves or formally gates the five mutation-required control fields
left by T007 before any controlled substitution experiment. It uses existing
serialized evidence and bounded read-only inspection of the pinned
`vllm==0.21.0` install.

No substitution was implemented. No vLLM source was modified. No monkeypatching,
allocator replacement, scheduler behavior change, worker layout change, Triton
kernel, copy kernel, LSDR, serving change, or quality evaluation was performed.

## Inputs

- `research/avmp/v2/results/vllm-planner-stage-observation/translated_planner_stage_config.json`
- `research/avmp/v2/results/vllm-planner-stage-advisory-diff/diff_report.json`
- `research/avmp/v2/results/vllm-mutation-readiness/readiness_report.json`
- `research/avmp/v2/PATH_C_MUTATION_HOOK_DESIGN_GATE.md`
- `research/avmp/v2/PATH_C_SHIM_AUDIT.md`
- Read-only grep inspection of
  `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm`

## Existing Evidence Summary

The current evidence is strong enough for advisory planning:

- `get_kv_cache_configs(...)` was reached with real planner inputs.
- Runtime scheduler parity was observed for the replayed planner output.
- The translated planner output has `num_blocks=329`, `cache_group_count=7`,
  `cache_tensor_count=9`, and `layer_count=63`.
- The advisory diff reports `vanilla_reserved_bytes=2910781440`,
  `vanilla_useful_bytes=1679258112`,
  `cachepawl_proposed_reserved_bytes=1679258112`,
  `estimated_savings_bytes=1231523328`,
  `overestimation_ratio=1.7333734577189286`, and
  `wasted_fraction=0.4230902777777778`.
- T007 passed planner schema, block/page size, dtype/state dtype,
  Mamba shape, attention/Mamba mapping, and estimated/useful byte checks.

The same evidence is not enough for controlled substitution because it does not
prove how a substituted plan would remain coherent across scheduler admission,
`KVCacheManager` state, worker tensor allocation, request block assignment,
Mamba state indices, and attention views.

## Field 1: Stable Scheduler Or Planner Construction Hook

### Current Evidence

`PATH_C_MUTATION_HOOK_DESIGN_GATE.md` selected post-call advisory/diff because
it can observe the planner boundary without changing vLLM behavior. The pinned
source exposes the planner functions:

- `vllm.v1.core.kv_cache_utils.get_kv_cache_configs(...)`
- `vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups(...)`
- `vllm.v1.core.kv_cache_utils.get_kv_cache_groups(...)`

Read-only source inspection also confirms that
`vllm.v1.core.sched.scheduler.Scheduler.__init__` directly constructs
`KVCacheManager(...)`.

### Existing Artifacts Resolve It

No. The artifacts resolve planner observation and replay parity. They do not
identify a stable public construction hook for returning a substituted
`KVCacheConfig` to vLLM or replacing scheduler construction without a
monkeypatch, source edit, local fork, or upstream seam.

### Exact vLLM Object/Path/Function To Observe

- `LLM.llm_engine.engine_core.engine_core.vllm_config`
- `LLM.llm_engine.engine_core.engine_core.model_executor.get_kv_cache_specs()`
- `LLM.llm_engine.engine_core.engine_core.available_gpu_memory_for_kv_cache`
- `vllm.v1.core.kv_cache_utils.get_kv_cache_configs(...)`
- `vllm.v1.core.sched.scheduler.Scheduler.__init__`
- `LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config`

### Can Read-Only Observation Resolve It

Only for advisory confidence. Read-only observation can continue to prove where
the planner input/output boundary is and whether vanilla replay remains stable.
It cannot create or validate a sanctioned substitution seam.

### Mutation Required

Yes, for controlled substitution. Returning a different planner result inside
vLLM would require an invasive control path unless vLLM exposes a factory,
callback, or extension point not found in the pinned source.

### Correctness Risk

High. A substituted planner result must satisfy scheduler, manager, worker
allocation, attention, and Mamba contracts at once.

### Recommendation

`hard_blocker_for_current_cycle`. Keep planner work advisory-only unless a
separate future decision selects a local fork or upstream extension seam.

## Field 2: Allocator Or KVCacheManager Replacement Control Point

### Current Evidence

`PATH_C_SHIM_AUDIT.md` records that `Scheduler.__init__` constructs
`KVCacheManager` directly and stores it at `self.kv_cache_manager`. The manager
delegates allocation behavior through coordinator and block-pool internals.
Read-only source inspection confirms:

- `vllm.v1.core.kv_cache_manager.KVCacheManager`
- `KVCacheManager.allocate_slots(...)`
- `KVCacheManager.get_block_ids(request_id)`
- `KVCacheManager.take_new_block_ids()`
- `KVCacheManager.block_pool`
- `vllm.v1.core.kv_cache_coordinator.HybridKVCacheCoordinator`
- `vllm.v1.core.single_type_kv_cache_manager.MambaManager`

### Existing Artifacts Resolve It

No. Existing artifacts observe final plans and advisory savings, but they do
not prove a no-monkeypatch replacement point for `KVCacheManager`, its
coordinator, or its `BlockPool`.

### Exact vLLM Object/Path/Function To Observe

- `LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager`
- `scheduler.kv_cache_manager.kv_cache_config`
- `scheduler.kv_cache_manager.block_pool`
- `scheduler.kv_cache_manager.get_usage()`
- `scheduler.kv_cache_manager.get_block_ids(request_id)`
- `scheduler.kv_cache_manager.take_new_block_ids()`

### Can Read-Only Observation Resolve It

No for replacement control. Read-only observation can characterize manager
state, block-pool usage, and request block ids, but it cannot prove replacement
safety or provide an injection mechanism.

### Mutation Required

Yes. Replacement would require scheduler construction control, a local fork,
an upstream factory seam, or monkeypatching. Monkeypatching is out of scope.

### Correctness Risk

High. `KVCacheManager`, coordinator, prefix-cache state, block pool, event
tracking, and request allocation are tightly coupled.

### Recommendation

`hard_blocker_for_current_cycle`. Do not attempt allocator or manager
replacement in T009.

## Field 3: Worker Tensor Allocation Layout Control Point

### Current Evidence

The existing artifacts quantify padded tensor over-reservation, but they do not
change worker allocation. The pinned source shows worker allocation is handled
inside `vllm.v1.worker.gpu_model_runner.GPUModelRunner`:

- `get_kv_cache_spec()`
- `_allocate_kv_cache_tensors(...)`
- `_reshape_kv_cache_tensors(...)`
- `_update_hybrid_attention_mamba_layout(...)`
- `initialize_kv_cache_tensors(...)`
- `initialize_kv_cache(...)`

`PATH_C_SHIM_AUDIT.md` records that `_reshape_kv_cache_tensors(...)` creates
attention and Mamba views and handles `page_size_padded` through strided views.

### Existing Artifacts Resolve It

No. They resolve the planner-level shape and byte accounting, not the worker
layout rewrite contract.

### Exact vLLM Object/Path/Function To Observe

- `GPUModelRunner.get_kv_cache_spec()`
- `GPUModelRunner.initialize_kv_cache(...)`
- `GPUModelRunner.initialize_kv_cache_tensors(...)`
- `GPUModelRunner._allocate_kv_cache_tensors(...)`
- `GPUModelRunner._reshape_kv_cache_tensors(...)`
- `GPUModelRunner._update_hybrid_attention_mamba_layout(...)`

### Can Read-Only Observation Resolve It

Partially. A bounded read-only capture can record raw tensor sizes, reshaped
view shapes/strides, attention cache views, Mamba cache views, and whether
hybrid layout rewriting runs. It still would not authorize changing the layout.

### Mutation Required

Yes, for memory-saving behavior. Reducing raw tensor allocation or changing
view layout would require a coherent substituted `KVCacheConfig` and worker
layout changes.

### Correctness Risk

Very high. Incorrect layout changes can break attention backend dimensions,
Mamba state tensors, strided views, and block-table metadata.

### Recommendation

`resolvable_by_read_only_observation` for documenting the vanilla layout
contract, but not for substitution. The next smallest useful task should remain
read-only and capture worker tensor layout metadata only.

## Field 4: Runtime Request-To-Block Assignment Control

### Current Evidence

The artifacts prove planner parity and final cache topology, but they do not
observe request-time block assignment. `PATH_C_SHIM_AUDIT.md` identifies
manager methods that expose request-to-block state without mutation.

### Existing Artifacts Resolve It

No. There is no artifact that records a live request id, allocated block ids,
new block ids, block-pool usage before/after a bounded request, or Mamba
manager per-request state.

### Exact vLLM Object/Path/Function To Observe

- `LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager`
- `KVCacheManager.allocate_slots(...)` call effects through surrounding
  scheduler output
- `KVCacheManager.get_block_ids(request_id)`
- `KVCacheManager.take_new_block_ids()`
- `KVCacheManager.get_usage()`
- `KVCacheManager.block_pool.get_num_free_blocks()`
- `KVCacheManager.take_events()`

### Can Read-Only Observation Resolve It

Yes for characterization. A bounded vanilla request can observe manager state
before and after scheduling without mutating vLLM. It cannot provide control.

### Mutation Required

Yes, for alternative assignment. Any change to assignment policy would require
manager/coordinator mutation and must preserve scheduler and prefix-cache
contracts.

### Correctness Risk

High. Request block assignment is coupled to allocation, prefix caching,
computed-block reuse, block-pool events, and scheduler outputs.

### Recommendation

`resolvable_by_read_only_observation` for evidence gathering. Do not treat this
as resolved for substitution until a read-only runtime assignment artifact
exists and a separate mutation seam has been accepted.

## Field 5: Mamba State-Index And Attention View Rewrite Contract

### Current Evidence

The existing artifacts verify Mamba shape and attention/Mamba group mapping at
the planner level. The pinned source shows runtime Mamba metadata is built in
`vllm.v1.attention.backends.mamba_attn`:

- `BaseMambaAttentionMetadataBuilder._compute_common_metadata(...)`
- `mamba_get_block_table_tensor(...)`
- `BaseMambaAttentionMetadataBuilder.update_block_table(...)`
- `state_indices_tensor_d`
- `state_indices_tensor_p`

Worker-side attention and Mamba views are created by
`GPUModelRunner._reshape_kv_cache_tensors(...)` and adjusted by
`GPUModelRunner._update_hybrid_attention_mamba_layout(...)`.

### Existing Artifacts Resolve It

No. Existing artifacts prove serialized shapes and groups, but not the runtime
state-index/view rewrite contract.

### Exact vLLM Object/Path/Function To Observe

- `vllm.v1.attention.backends.mamba_attn.BaseMambaAttentionMetadataBuilder._compute_common_metadata(...)`
- `vllm.v1.attention.backends.mamba_attn.mamba_get_block_table_tensor(...)`
- `vllm.v1.attention.backends.mamba_attn.BaseMambaAttentionMetadataBuilder.update_block_table(...)`
- `GPUModelRunner._reshape_kv_cache_tensors(...)`
- `GPUModelRunner._update_hybrid_attention_mamba_layout(...)`
- Runtime block-table tensors and `state_indices_tensor_d/p` shapes

### Can Read-Only Observation Resolve It

Partially. Read-only observation can record vanilla block-table tensor shapes,
Mamba state-index tensor shapes, and attention/Mamba view shapes/strides during
a bounded request. It cannot prove a rewritten layout is valid.

### Mutation Required

Yes, for AVMP layout substitution. A rewritten layout would need to update
block tables, Mamba state indices, attention views, and any affected metadata
builders coherently.

### Correctness Risk

Very high. This is the highest-risk field because a shape-compatible plan can
still fail if Mamba state indices or attention views point at the wrong block
or stride.

### Recommendation

`resolvable_by_read_only_observation` for vanilla contract capture; otherwise a
`hard_blocker_for_current_cycle` for controlled substitution.

## Control-Field Classification

| Field | Classification | Reason |
|---|---|---|
| stable scheduler or planner construction hook | `hard_blocker_for_current_cycle` | Planner replay is available, but no stable public substitution seam was found. |
| allocator or KVCacheManager replacement control point | `hard_blocker_for_current_cycle` | `Scheduler.__init__` constructs `KVCacheManager` directly; replacement requires mutation outside this cycle's safety boundary. |
| worker tensor allocation layout control point | `resolvable_by_read_only_observation` | Vanilla allocation/view layout can be captured, but changing it remains out of scope. |
| runtime request-to-block assignment control | `resolvable_by_read_only_observation` | Live assignment can be observed through manager methods, but control requires manager/coordinator mutation. |
| Mamba state-index and attention view rewrite contract | `resolvable_by_read_only_observation` | Vanilla block-table/state-index/view contracts can be captured, but rewrites remain too risky for this cycle. |

## Go/No-Go Result

Final classification: `stay_advisory_only_this_cycle`.

Do not open T009 as a controlled substitution experiment. The two core control
fields required to safely return or enforce a changed allocation plan are hard
blocked for this cycle, and three runtime-contract fields still need bounded
read-only observation before a future cycle can reconsider mutation.

## Smallest Bounded Next Task

Open the next task as read-only runtime contract observation, not substitution.

Recommended scope:

- Capture a bounded vanilla runtime allocation/metadata artifact in the pinned
  vLLM environment.
- Observe `scheduler.kv_cache_manager` before and after one bounded request.
- Record `KVCacheManager.get_usage()`, free-block count, request block ids when
  available, and new block ids/events when safely observable.
- Record worker KV tensor raw sizes plus reshaped attention/Mamba view
  shapes/strides if available without monkeypatching.
- Record Mamba block-table and `state_indices_tensor_d/p` shapes if reachable
  without modifying vLLM.
- Preserve no source edits, no monkeypatching, no allocator replacement, no
  scheduler mutation, no worker layout mutation, and no quality evaluation.

Stop immediately if any observation requires monkeypatching, replacing vLLM
objects, changing scheduler behavior, or changing worker allocation.

## Future Controlled-Substitution Requirements

If a later cycle reopens controlled substitution, it must require:

- default-off feature flag
- explicit opt-in for every substituted run
- no mutation in normal advisory mode
- persisted vanilla planner output, Cachepawl proposed output, and substituted
  output
- validation before returning any substituted config to vLLM
- rollback to vanilla planner output on any validation failure
- no partially mutated runtime state after failure
- bounded parity artifact using the same model, vLLM version, runtime bounds,
  and durable environment
- explicit proof that scheduler, manager, worker layout, Mamba state indices,
  and attention views remain coherent
- separate decision if the selected path requires a local vLLM fork or upstream
  extension seam

