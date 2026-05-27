# Evaluation

## Setup

The evaluation uses existing observe/advisory artifacts from a vanilla
`vllm==0.21.0` run of `Zyphra/Zamba2-2.7B-instruct`.

Configuration:

- Max model lengths: `2048`, `4096`
- Max number of sequences: `1`
- GPU memory utilization values: `0.6`, `0.7`
- Durable vLLM environment:
  `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv`
- Observed tensor device: `cuda:0`

The artifacts do not record a GPU model name. We therefore report only the
device identifier captured in safe tensor metadata.

## Method

The evaluation has five stages:

1. Observe the vanilla vLLM runtime cache plan and safe runtime metadata.
2. Translate the vLLM cache plan into Cachepawl's schema.
3. Replay `vllm.v1.core.kv_cache_utils.get_kv_cache_configs` on real planner
   inputs in a bounded post-initialization run.
4. Run `cachepawl diagnose-vllm` on artifact inputs to produce advisory metrics.
5. Repeat the planner-stage observation and advisory diff over the bounded
   4-cell advisory matrix.

All runtime observations were read-only. The workflow did not modify vLLM,
monkeypatch private methods, replace allocators, return Cachepawl plans to vLLM,
alter scheduler behavior, or alter worker tensor layout.

## Planner Replay Result

Planner-stage replay succeeded and produced `planner_stage_translation`. The
translated planner output matched the runtime scheduler cache config:

- `planner_matches_runtime_scheduler=true`
- `runtime_changed_during_replay=false`

This establishes that the advisory comparison is grounded in the same cache
configuration used by the runtime scheduler for each observed planner-stage
matrix cell.

## Advisory Metrics

| max_model_len | gpu_memory_utilization | max_num_seqs | estimated advisory savings bytes | num_blocks |
| ---: | ---: | ---: | ---: | ---: |
| `2048` | `0.6` | `1` | `801,051,648` | `214` |
| `2048` | `0.7` | `1` | `1,347,563,520` | `360` |
| `4096` | `0.6` | `1` | `685,011,456` | `183` |
| `4096` | `0.7` | `1` | `1,231,523,328` | `329` |

Across all four completed cells, `overestimation_ratio` stayed
`1.7333734577189286` and `wasted_fraction` stayed `0.4230902777777778`.
Estimated advisory savings ranged from `685,011,456` to `1,347,563,520` bytes.

The estimated savings are planner-level advisory savings. They are not measured
runtime VRAM reductions, and they do not support latency, throughput, or quality
claims.

## Runtime Contract Observations

The runtime contract observations resolved several mutation-readiness questions:

- Scheduler/KVCacheManager structure was observed.
- Block usage metadata was observed.
- Worker tensor layout was observed. The runtime contract artifact captured 32
  worker tensor summaries; the first had shape `[2, 329, 48, 32, 160]`.
- Live request-to-block assignment was observed. A bounded request saw active
  block ids `[1, 2, 3, 4, 5, 6, 7]` after the first scheduler step.
- Attention block-table/view metadata was observed, with 21 block-table tensor
  metadata summaries.
- Attention metadata builders were observed, with 7 attention groups.

The Mamba side remains gated:

- `mamba_state_index_contract` is blocked because `mamba_state_idx` was
  reachable but empty for the live request.
- `mamba_state_tensor_contract` is blocked because no Mamba state tensors were
  safely reachable by stable runtime attributes.
- Runtime cache config reported `mamba_cache_mode: none`.

## Product Artifact

`cachepawl diagnose-vllm` is the productized output of this phase. It consumes
translated cache-plan artifacts and emits `report.json`, `summary.md`, and
`manifest.json`. It is designed for advisory use in environments that do not
have vLLM, CUDA, GPU access, or NVML installed.

## Interpretation

The evaluation supports a diagnostic claim: Cachepawl can expose and report
planner-level cache overestimation for a real hybrid vLLM cache plan across a
small bounded config matrix. It remains a single-model advisory result. It does
not support a mutation claim. Controlled substitution requires stronger Mamba
state-index and state tensor contracts, or a supported upstream integration seam.
