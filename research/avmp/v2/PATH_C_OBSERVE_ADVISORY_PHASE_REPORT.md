# Path C Observe/Advisory Phase Report

Date: 2026-05-27

Scope: close the vLLM Path C observe/advisory phase for
`Zyphra/Zamba2-2.7B-instruct` on vanilla `vllm==0.21.0`.

Conclusion: the advisory/diagnostic path is valid and useful, but controlled
substitution is not approved for this cycle. Future mutation work requires a
run/model/config where Mamba state-index and Mamba state tensors are observable,
or a supported vLLM integration seam that defines those rewrite contracts.

## Evidence Chain

1. Runtime cache-plan observation reached vanilla vLLM runtime objects without
   modifying vLLM, monkeypatching, replacing allocators, or returning Cachepawl
   plans.
2. Planner-stage replay reached the real planner inputs and called
   `vllm.v1.core.kv_cache_utils.get_kv_cache_configs` in a bounded replay.
   The replay status was `planner_stage_translation`.
3. Planner output matched the runtime scheduler cache config. The replay did
   not change the runtime scheduler config:
   `runtime_changed_during_replay=false`.
4. Planner-stage advisory diff produced an advisory-only savings estimate from
   the observed vLLM cache plan.
5. Mutation-readiness checks passed all structural invariants but stayed
   `advisory_only_recommended` because mutation-control contracts were not all
   resolved.
6. Runtime contract observation resolved scheduler/KV manager structure, block
   usage metadata, and worker tensor layout, while identifying live
   request/block and Mamba/attention contracts as remaining gaps.
7. Live-request observation resolved request-to-block assignment for a single
   bounded vanilla vLLM request.
8. Mamba/attention contract observation resolved the attention-side contracts
   and formally gated the remaining Mamba state contracts.

## Key Metrics

- Model: `Zyphra/Zamba2-2.7B-instruct`
- vLLM version: `0.21.0`
- Max model length: `4096`
- Max number of sequences: `1`
- GPU memory utilization: `0.7`
- Cache groups: `7`
- Cache tensors: `9`
- Layers covered: `63`
- Num blocks: `329`
- Vanilla reserved bytes: `2,910,781,440`
- Vanilla useful bytes: `1,679,258,112`
- Cachepawl proposed reserved bytes: `1,679,258,112`
- Estimated advisory savings bytes: `1,231,523,328`
- Overestimation ratio: `1.7333734577189286`
- Wasted fraction: `0.4230902777777778`

## Resolved Contracts

- Planner schema compatibility: passed.
- Num blocks compatibility: passed.
- Cache group count compatibility: passed.
- Cache tensor count compatibility: passed.
- Layer coverage compatibility: passed.
- Dtype and state dtype compatibility: passed.
- Block and page size compatibility: passed.
- Mamba state shape compatibility: passed.
- Attention/Mamba group mapping compatibility: passed.
- Estimated/useful byte consistency: passed.
- Scheduler/KVCacheManager structure: observed.
- Block usage metadata: observed.
- Worker cache tensor layout: observed.
- Runtime request-to-block assignment: observed.
- Attention block-table/view contract: observed.
- Attention metadata builder contract: observed.

The live-request observation saw scheduler request id `0-aaa14650`, completion
output request id `0`, active block ids `[1, 2, 3, 4, 5, 6, 7]` after the first
step, and block-pool usage return from `321 / 329` free GPU blocks during the
request to `328 / 329` after completion.

The Mamba/attention observation saw request id `0-a1ee70e3`, `21` block-table
tensor metadata summaries, and `7` attention groups. No tensor payloads or large
model objects were serialized.

## Remaining Blockers

- `mamba_state_index_contract`: blocked. `mamba_state_idx` was reachable on the
  GPU model runner, but it remained empty for the live request.
- `mamba_state_tensor_contract`: blocked. No Mamba state tensors were safely
  reachable through stable runtime attributes in the bounded vanilla run.
- Runtime cache config reported `mamba_cache_mode: none`, so this run did not
  populate the Mamba state-index/state paths needed to define a rewrite
  contract.

These blockers are not evidence that Cachepawl cannot support Mamba state
rewrites. They are evidence that this specific vanilla vLLM run does not expose
the required state-index and state-tensor contracts safely enough to approve
mutation.

## Why Controlled Substitution Is Not Allowed

Controlled substitution would require Cachepawl to rewrite or replace runtime
cache views while preserving vLLM's request-to-block assignment, attention block
tables, Mamba state indices, and Mamba state tensors. The attention side is now
observable, but the Mamba state side is not.

Approving mutation without the Mamba state-index and state-tensor contracts
would require one of the disallowed actions: modifying vLLM source,
monkeypatching private execution paths, replacing allocators without a supported
seam, or inferring state layout from incomplete runtime evidence. This phase
therefore remains advisory-only.

## Future Mutation Requirements

A future mutation cycle needs at least one of the following:

- A vanilla vLLM run/model/config where `mamba_cache_mode` causes Mamba
  state-index entries and Mamba state tensors to be populated and safely
  reachable during a bounded live request.
- A supported vLLM integration seam that defines how an external allocator may
  provide or rewrite Mamba state views and attention block tables.
- A documented, default-off mutation probe that is explicitly approved after
  the above contracts are observed or supported by upstream interfaces.

Before any substitution experiment, the project must also keep the existing
controls: feature flag default-off, bounded workload, rollback path, no serving
changes, no quality evaluation coupling, and no allocator replacement outside
the approved seam.

## Current Product Value

`cachepawl diagnose-vllm` is the valid product surface for this cycle. It turns
observed vLLM cache-plan artifacts into an advisory report without requiring the
main environment to import vLLM, use CUDA, use NVML, modify vLLM, monkeypatch,
replace allocators, or return plans to vLLM.

The committed evidence supports an advisory value proposition: Cachepawl can
identify cache-plan overestimation for hybrid workloads and report concrete
planner-level memory savings while remaining non-invasive.

## Recommended Next Product Task

Package the observe/advisory path as the primary Path C deliverable:

- Add a concise user-facing guide for collecting vLLM observation artifacts in
  the durable vLLM environment.
- Document how to run `cachepawl diagnose-vllm` from artifact inputs.
- Preserve the current advisory-only classification in examples.
- Keep mutation work out of the product path until Mamba state contracts are
  observable or supported by vLLM.
