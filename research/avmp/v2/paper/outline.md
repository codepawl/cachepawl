# Cachepawl vLLM Path C Observe/Advisory Paper Outline

## Working Title

Cachepawl Path C: Advisory Diagnosis of Hybrid Attention/Mamba Cache
Overestimation in vLLM

## Positioning

This paper presents a diagnostic/advisory systems artifact. It studies a real
vanilla `vllm==0.21.0` hybrid cache planner-stage replay for
`Zyphra/Zamba2-2.7B-instruct` and packages the resulting analysis as
`cachepawl diagnose-vllm`.

The paper does not claim runtime mutation, allocator replacement, serving-time
VRAM reduction, throughput improvement, latency improvement, or quality impact.

## 1. Introduction

- Hybrid models combine attention layers and Mamba/SSM state layers.
- Existing inference cache planners can reserve memory according to uniform
  page/block assumptions that overestimate useful cache bytes for hybrid
  layouts.
- Cachepawl Path C asks whether a non-invasive observe/advisory workflow can
  expose that overestimation in a real vLLM runtime.
- Main result: across a bounded 4-cell advisory matrix for one model,
  estimated savings range from `685,011,456` to `1,347,563,520` bytes while
  `overestimation_ratio=1.7333734577189286` and
  `wasted_fraction=0.4230902777777778` remain constant.

## 2. Problem Statement

- Attention KV cache and Mamba state cache have different shapes, lifetimes,
  and page economics.
- A uniform cache-reservation path can hide wasted capacity behind runtime
  planner abstractions.
- Direct mutation is unsafe without contracts for scheduler construction,
  request-to-block assignment, worker layout, attention block tables, and Mamba
  state-index/state tensor views.

## 3. Method

- Observe: capture vLLM runtime cache-plan artifacts and safe runtime metadata.
- Translate: normalize vLLM cache plans into Cachepawl's schema.
- Replay: call the vLLM planner stage on real planner inputs in a bounded,
  post-initialization run.
- Diagnose: compute advisory reserved/useful/savings metrics and emit an
  artifact-input CLI report.
- Gate: inspect runtime contracts before approving any controlled substitution.

## 4. System Artifact

- `cachepawl diagnose-vllm` consumes existing translated vLLM cache artifacts.
- The CLI does not import vLLM, require CUDA/NVML, load a model, modify vLLM,
  monkeypatch, replace allocators, or return Cachepawl plans to vLLM.
- Output is an advisory report with planner-level memory metrics and explicit
  mutation blockers.

## 5. Evaluation

- Model: `Zyphra/Zamba2-2.7B-instruct`
- Runtime: vanilla `vllm==0.21.0`
- Workload bounds: one sequence, `max_model_len` in `{2048, 4096}`,
  `gpu_memory_utilization` in `{0.6, 0.7}`
- Matrix result:
  - `2048 / 0.6`: `801,051,648` bytes advisory savings;
  - `2048 / 0.7`: `1,347,563,520` bytes advisory savings;
  - `4096 / 0.6`: `685,011,456` bytes advisory savings;
  - `4096 / 0.7`: `1,231,523,328` bytes advisory savings.
- Evidence:
  - planner-stage replay matched the runtime scheduler config;
  - `runtime_changed_during_replay=false`;
  - live request block assignment observed;
  - worker tensor layout observed;
  - attention block-table/view metadata observed;
  - attention metadata builders observed;
  - Mamba state-index and state tensor contracts remain blocked.

## 6. Limitations

- No runtime mutation claim.
- No runtime VRAM, throughput, latency, or quality claim.
- Single model with four bounded config cells; no cross-model or workload
  generalization claim.
- Mamba state contracts are unresolved because `mamba_state_idx` was reachable
  but empty and no Mamba state tensors were safely reachable.
- Observed runtime cache config reported `mamba_cache_mode: none`.

## 7. Future Work

- Find a model/config/run where Mamba state-index and Mamba state tensors are
  observable through safe runtime paths.
- Define or use a supported vLLM integration seam for controlled substitution.
- Expand to multi-model and multi-workload evaluation.
- Only attempt runtime substitution after stronger contracts and default-off
  rollback controls are in place.

## 8. Conclusion

Cachepawl Path C demonstrates that a non-invasive observe/translate/replay/
diagnose workflow can expose hybrid cache overestimation in real vLLM planner
artifacts. The result is useful as an advisory diagnostic tool today, while the
system correctly refuses controlled substitution until Mamba state contracts are
observable or supported by vLLM.
