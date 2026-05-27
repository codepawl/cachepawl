# Abstract

Hybrid Attention/Mamba language models stress inference cache planners because
attention KV pages and Mamba state caches have different shapes and reuse
contracts. Cachepawl Path C evaluates whether these planner-level inefficiencies
can be diagnosed in a real vLLM runtime without modifying the serving stack.

We present an observe/advisory workflow that captures vanilla `vllm==0.21.0`
cache artifacts for `Zyphra/Zamba2-2.7B-instruct`, translates them into a
Cachepawl schema, replays the vLLM planner stage on real planner inputs, and
emits an advisory report through `cachepawl diagnose-vllm`. The planner-stage
replay matched the runtime scheduler cache configuration and did not change the
runtime scheduler state. Across a bounded four-cell matrix for one model
(`max_model_len` in `{2048, 4096}`, `gpu_memory_utilization` in `{0.6, 0.7}`,
and `max_num_seqs=1`), estimated advisory savings ranged from `685,011,456` to
`1,347,563,520` bytes. The overestimation ratio stayed
`1.7333734577189286x` and the wasted fraction stayed `42.3%` across completed
cells.

The artifact also records runtime contracts needed before any mutation attempt.
Live request-to-block assignment, worker tensor layout, attention block-table
views, and attention metadata builders were observed. Mamba state-index and
Mamba state tensor contracts remain blocked: `mamba_state_idx` was reachable but
empty for the live request, no Mamba state tensors were safely reachable, and
the observed cache config used `mamba_cache_mode: none`.

This is a diagnostic/advisory systems artifact. It does not claim runtime cache
substitution, allocator replacement, serving-time VRAM reduction, throughput
improvement, latency improvement, or quality impact. Its contribution is a
non-invasive method and product surface for finding hybrid cache overestimation
while preserving explicit gates for future controlled substitution.
