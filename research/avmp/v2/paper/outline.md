# Cachepawl vLLM Path C Observe/Advisory Technical Report Plan

## Working Title

Cachepawl Path C: Advisory Diagnosis of Hybrid Attention/Mamba Cache
Overestimation in vLLM

## Report Positioning

This report is a technical artifact paper for the observe/advisory phase of
Cachepawl's vLLM Path C work. It describes how Cachepawl observes a vanilla
`vllm==0.21.0` hybrid cache plan, translates the resulting artifacts, replays
the planner stage, and emits a user-facing advisory report through
`cachepawl diagnose-vllm`.

The report's central claim is deliberately narrow: a non-invasive
observe/translate/replay/diagnose workflow can expose planner-level cache
overestimation in an existing hybrid vLLM cache plan while preserving explicit
gates against unsafe runtime mutation.

The report does not claim:

- runtime VRAM reduction,
- throughput improvement,
- latency improvement,
- model quality or accuracy improvement,
- allocator replacement,
- vLLM modification,
- controlled substitution readiness.

## Draft Components

### 1. Abstract

Use `abstract.md`.

The abstract should state the advisory result and the negative scope in the
same paragraph: this is planner-level diagnosis, not serving-time substitution.

### 2. Introduction

Use `introduction.md`.

Purpose:

- explain why hybrid Attention/Mamba models stress uniform cache planning;
- introduce Path C as an observe/advisory phase rather than a mutation phase;
- state the exact artifact claim;
- summarize the bounded 4-cell matrix result:
  - `overestimation_ratio=1.7333734577189286`;
  - `wasted_fraction=0.4230902777777778`;
  - advisory savings range from `685,011,456` to `1,347,563,520` bytes.

### 3. Method

Use `method.md`.

Structure:

1. Capture vanilla vLLM runtime cache-plan artifacts.
2. Translate the cache plan into Cachepawl's schema.
3. Replay the vLLM planner stage on observed planner inputs.
4. Compute advisory reserved/useful/savings metrics.
5. Package the result through `cachepawl diagnose-vllm`.
6. Inspect runtime contracts that would be required before mutation.

The method section should explicitly say that artifact-input diagnosis does not
import vLLM, load a model, require GPU/NVML, modify vLLM, replace allocators, or
return Cachepawl plans to vLLM.

Problem framing:

- hybrid Attention/Mamba cache planning can reserve substantially more memory
  than useful cache payloads;
- the report evaluates whether this over-reservation can be diagnosed from
  artifacts before any mutation path is considered.

### 4. Evaluation

Use `evaluation_section.md`.

Evidence to include:

- one model: `Zyphra/Zamba2-2.7B-instruct`;
- vanilla `vllm==0.21.0`;
- four cells:
  - `max_model_len` in `{2048, 4096}`;
  - `gpu_memory_utilization` in `{0.6, 0.7}`;
  - `max_num_seqs=1`;
- planner-stage replay matched the runtime scheduler config;
- `runtime_changed_during_replay=false`;
- request-to-block assignment was observed;
- worker tensor layout was observed;
- attention block-table/view metadata was observed;
- Mamba state-index/tensor contracts remained blocked.

### 5. Artifact Appendix

Use `artifact_appendix.md`.

Purpose:

- map claims to committed artifact paths;
- identify which artifacts support advisory metrics;
- identify which artifacts support contract observations;
- list generated CLI outputs: `report.json`, `summary.md`, `manifest.json`.

### 6. Limitations

Use `limitations.md`.

The limitations section is part of the core claim, not an afterthought. It
should make the blocked contracts and non-claims impossible to miss.

### 7. Conclusion

The conclusion should close on the artifact boundary: Path C produced a useful
diagnostic surface and evidence package, while correctly refusing runtime
mutation until Mamba state-index and state tensor contracts are observable or
vLLM exposes a supported integration path.

## Claim Checklist

Allowed claims:

- 4-cell advisory matrix completed.
- Stable `overestimation_ratio=1.7333734577189286`.
- Stable `wasted_fraction=0.4230902777777778`.
- Advisory savings range from `685,011,456` to `1,347,563,520` bytes.
- Planner-stage replay matched runtime scheduler config.
- `runtime_changed_during_replay=false`.
- Request-to-block assignment was observed.
- Worker tensor layout was observed.
- Attention block-table/view metadata was observed.
- Mamba state-index and tensor contracts were blocked.

Disallowed claims:

- Runtime VRAM reduction.
- Runtime throughput, latency, or accuracy improvement.
- Runtime cache substitution.
- Allocator replacement.
- vLLM modification.
- Generalization beyond the bounded model/config evidence.
