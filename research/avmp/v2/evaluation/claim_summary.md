# Paper Claim Summary

This table is the paper-facing claim boundary for the current vLLM Path C
observe/advisory evidence pack. It is a summary of committed artifacts, not a
new benchmark result.

| Claim | Status | Evidence | Paper Use |
| --- | --- | --- | --- |
| Cachepawl can consume translated vLLM cache-plan artifacts and emit an advisory report without importing vLLM in the package environment. | Supported | `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/report.json`, `summary.md`, `manifest.json`; `tests/cli/test_diagnose_vllm.py` | Product artifact and reproducibility claim. |
| Planner-stage replay matched the runtime scheduler cache config for the observed cells. | Supported | `research/avmp/v2/results/vllm-planner-stage-observation/translated_planner_stage_config.json`; `research/avmp/v2/evaluation/artifact_index.md` | Method validity for planner-level advisory comparison. |
| The bounded 4-cell matrix reports planner-level advisory over-reservation for `Zyphra/Zamba2-2.7B-instruct` on `vllm==0.21.0`. | Supported | `research/avmp/v2/evaluation/matrix_table.md`; `matrix_table.csv` | Main numeric result. Report advisory savings range `685,011,456` to `1,347,563,520` bytes, overestimation ratio `1.7333734577189286`, and wasted fraction `0.4230902777777778`. |
| Runtime observations are read-only and do not mutate vLLM scheduler, allocator, or worker tensor state. | Supported | `research/avmp/v2/evaluation/methodology.md`; runtime contract artifacts listed in `artifact_index.md` | Non-mutation control. |
| Controlled runtime substitution is ready. | Not supported | `research/avmp/v2/evaluation/limitations.md`; `research/avmp/v2/results/vllm-mutation-readiness/readiness_report.json` | State as limitation. Mamba state-index and state tensor contracts remain blocked. |
| Runtime VRAM, latency, throughput, quality, or accuracy improvement has been measured for this Path C advisory CLI phase. | Not measured | `research/avmp/v2/evaluation/limitations.md`; `research/avmp/v2/evaluation/methodology.md` | Do not claim. Future work only. |

## Paper-Ready Result Statement

In a bounded observe/advisory evaluation on `Zyphra/Zamba2-2.7B-instruct` with
vanilla `vllm==0.21.0`, `max_model_len` in `{2048, 4096}`,
`gpu_memory_utilization` in `{0.6, 0.7}`, and `max_num_seqs=1`, Cachepawl's
artifact-input diagnostic workflow reported planner-level advisory savings from
`685,011,456` to `1,347,563,520` bytes. Across the four completed cells, the
planner-level overestimation ratio was `1.7333734577189286` and the wasted
fraction was `0.4230902777777778`.

These are advisory metrics over observed planner artifacts. They are not
measured runtime VRAM reductions and do not support serving throughput, latency,
model quality, or allocator-replacement claims.
