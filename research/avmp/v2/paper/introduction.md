# Introduction

Hybrid Attention/Mamba language models place two different cache shapes behind
one inference runtime. Attention layers use KV blocks whose economics are tied
to token sequence growth, while Mamba or SSM layers use state tensors whose
useful byte footprint can diverge from the page shape used by a uniform cache
planner. A cache planner that reserves by a shared page/block abstraction can
therefore allocate capacity according to a shape that is larger than the useful
state payload for hybrid layouts.

Cachepawl Path C studies this gap as an observe/advisory problem. The goal of
this phase is not to replace vLLM's allocator or to rewrite live runtime state.
Instead, the goal is to answer a narrower systems question: can Cachepawl
observe a real vanilla vLLM hybrid cache plan, translate it into a stable
schema, replay the planner stage on the observed inputs, and emit a practical
diagnostic report that quantifies planner-level overestimation?

The answer from the current artifact set is yes, within a bounded scope. The
evaluation uses `Zyphra/Zamba2-2.7B-instruct` under vanilla `vllm==0.21.0` and a
4-cell advisory matrix with `max_model_len` in `{2048, 4096}`,
`gpu_memory_utilization` in `{0.6, 0.7}`, and `max_num_seqs=1`. Across all four
completed cells, the planner-level `overestimation_ratio` stayed
`1.7333734577189286` and the planner-level `wasted_fraction` stayed
`0.4230902777777778`. Estimated advisory savings ranged from `685,011,456` to
`1,347,563,520` bytes.

Those values are advisory metrics over planner artifacts. They are not runtime
VRAM measurements, throughput measurements, latency measurements, or quality
measurements. The phase deliberately keeps artifact-input diagnosis separate
from controlled substitution.

The same evidence package also records what would be needed before mutation
could be considered. Planner-stage replay matched the runtime scheduler cache
configuration. Runtime observation resolved request-to-block assignment, worker
tensor layout, and attention block-table/view metadata. The Mamba side remains
blocked: state-index and state tensor contracts were not safely observable in
the current run. This makes the artifact useful in two ways: it provides a
diagnostic CLI users can run from committed artifacts, and it records why
allocator replacement is not claimed for this phase.

The contribution of this report is therefore an observe/advisory systems
artifact:

- a bounded method for capturing and translating vLLM hybrid cache-plan
  artifacts;
- planner-stage replay evidence that matches the runtime scheduler config;
- a user-facing `cachepawl diagnose-vllm` artifact-input report;
- a 4-cell advisory matrix with stable overestimation and waste metrics;
- an explicit contract checklist that blocks runtime mutation until Mamba
  state-index and state tensor paths are observable or supported upstream.
