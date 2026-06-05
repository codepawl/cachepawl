# Minimal vLLM Mamba State Contract Proposal

Status: proposal only. No vLLM source was modified.

## Problem

Cachepawl can observe stock vLLM scheduler/KV manager structure, worker cache
tensors, attention block tables, attention metadata builders, and live
request-to-block mappings. The missing runtime contract is the Mamba
state-index and Mamba state tensor view needed to prove that an alternate
hybrid cache plan can preserve the model's state semantics.

Observed blocker from
`research/avmp/v2/results/vllm-mamba-attention-contract-observation/mamba_attention_contract_report.json`:

- `mamba_state_index_contract`: blocked, `mamba_state_idx` was not reachable
  with live request id `0-a1ee70e3`
- `mamba_state_tensor_contract`: blocked, no Mamba state tensors were safely
  reachable by stable runtime attributes
- Attention block-table tensors and attention metadata groups were reachable,
  so the missing contract is Mamba-specific rather than total worker opacity

## Minimal Contract

Expose a read-only worker-runner method that returns a safe shape/index
snapshot for one request. The method must not serialize tensor contents and
must not mutate scheduler, worker, allocator, or cache state.

```python
class MambaStateContract(TypedDict):
    request_id: str
    state_index_present: bool
    state_index_value: int | None
    state_index_owner_path: str | None
    state_tensors: list[TensorSummary]
    attention_block_tables: list[TensorSummary]
    cache_mode: str
    block_size: int
    mamba_block_size: int | None

class TensorSummary(TypedDict):
    name: str
    owner_path: str
    shape: tuple[int, ...]
    stride: tuple[int, ...] | None
    dtype: str
    device: str
    layout: str | None
```

Candidate method:

```python
class GPUModelRunner:
    def get_mamba_state_contract(self, request_id: str) -> MambaStateContract:
        ...
```

## Required Semantics

- Return only scalar metadata and tensor shape/stride/dtype/device summaries.
- Include the stable owner path for `mamba_state_idx` or equivalent request to
  state-index mapping.
- Include the Mamba state tensor views used during decode for that request.
- Include attention block-table summaries from the same request phase so a
  caller can verify attention and Mamba views are aligned.
- Work before first decode step, after first decode step, and after
  completion, or explicitly report phase-specific absence.
- Return an explicit `cache_mode` value, including `none`, `align`, or `all`.
- Be safe under prefix caching on and off.

## Default-Off Substitution Gate

A Cachepawl micro-substitution probe remains blocked until this contract, or an
equivalent local-fork seam, is observable. Once observable, the first mutation
probe should be:

- default-off behind a single environment flag or CLI flag;
- one request only;
- no long-lived serving;
- rollback to vanilla plan before process exit;
- output parity checked against stock vLLM for the same prompt and sampling
  parameters;
- live GPU memory, admission/block assignment, latency, throughput, and output
  recorded only if substitution actually runs.

## Why Scheduler-Only Mutation Is Not Enough

The live-request artifact proves that vLLM can map request ids to block ids.
That is necessary but insufficient. Mamba state tensors have a different view
contract from attention block tables, and request block ids alone do not show
which Mamba state index or tensor view must be rewritten for an alternate
hybrid cache plan.

## Next Experiment

Run the GPU command matrix in `gpu_machine_commands.sh`, especially:

- `mamba_cache_mode=align`
- `mamba_cache_mode=all`
- prefix caching on and off
- shorter `max_model_len=1024` and `2048`
- `max_num_seqs=1` and `4`

If either `align` or `all` exposes Mamba state-index and state tensors, update
the readiness classification from blocker to implementation-ready and design
the default-off parity probe. If they do not, this proposal becomes the exact
upstream/local-fork integration seam.
