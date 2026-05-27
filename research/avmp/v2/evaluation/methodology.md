# Methodology

This evaluation pack summarizes a bounded observe/advisory phase. It does not
measure runtime mutation, runtime memory savings, latency, throughput, or model
quality.

## Environment

- Model: `Zyphra/Zamba2-2.7B-instruct`
- Runtime: vanilla `vllm==0.21.0`
- Durable vLLM environment: `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv`
- Python executable: `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv/bin/python`
- Max model lengths: `2048`, `4096`
- Max number of sequences: `1`
- GPU memory utilization values: `0.6`, `0.7`
- Observed tensor device: `cuda:0`

The artifacts do not record a GPU model name, so this pack reports only the
device identifier that appears in the safe tensor metadata.

## Procedure

1. Capture the vanilla vLLM runtime cache plan.
2. Replay the vLLM planner stage against real planner inputs in a bounded
   post-initialization run.
3. Translate the planner output into Cachepawl's vLLM cache-plan schema.
4. Compare vanilla reserved bytes with useful/proposed bytes to produce an
   advisory diff.
5. Repeat the planner-stage observation and advisory diff over the bounded
   4-cell config matrix: `max_model_len` in `{2048, 4096}`,
   `gpu_memory_utilization` in `{0.6, 0.7}`, and `max_num_seqs=1`.
6. Run mutation-readiness checks over the planner/runtime artifacts.
7. Observe runtime contracts without modifying vLLM:
   scheduler/KVCacheManager structure, block usage, worker cache tensor layout,
   live request-to-block assignment, attention block tables, attention metadata
   builders, and Mamba state paths where safely reachable.
8. Consolidate the result into an advisory-only phase report.

## Non-Mutation Controls

The observation phase did not:

- modify vLLM source,
- monkeypatch vLLM,
- replace allocators,
- return Cachepawl plans to vLLM,
- alter scheduler behavior,
- alter worker tensor layout,
- add runtime serving changes,
- add Triton kernels, copy kernels, or LSDR,
- run quality evaluation.

Only scalar metadata, JSON-safe records, and tensor shape/stride/dtype/device
summaries were captured.
