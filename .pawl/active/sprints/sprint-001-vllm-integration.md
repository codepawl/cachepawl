# Sprint 1 — vLLM integration baseline and shim

Status: In Progress
Created: 2026-05-23
Updated: 2026-05-25
Completed: N/A
TTL: 30 days after completion or cancellation
Archive After: N/A
Archive Warning: N/A
Archive Reason: N/A

## Goal

Prove cachepawl's Python AVMP allocator path inside vLLM by first capturing a vanilla baseline, then wiring the planned AVMP shim against the pinned vLLM version.

## Tasks

- [ ] `.pawl/active/tasks/t001-vllm-baseline-and-shim.md`

## Definition of Done

- [ ] vLLM development environment is documented and reproducible
- [ ] Vanilla vLLM baseline numbers are captured under `research/avmp/v2/results/`
- [ ] AVMP integration path is implemented or the fallback path is documented with evidence
- [x] New integration skeleton has focused tests and documented local verification
- [x] Planner-only baseline measurement spine has focused tests and documented local verification
- [x] Planner comparison evidence exists for vLLM-style padded planning versus Cachepawl AVMP
- [x] Planner benchmark memory metrics have unambiguous ratio and fraction names
- [x] RTX 3060 planner-comparison artifact pack is generated and documented
- [x] Vanilla vLLM baseline capture path records current not-runnable blocker
- [x] Local WSL2 GPU/NVML repair path is recorded as the runtime baseline infrastructure decision
- [x] Isolated pinned vLLM environment imports vLLM and sees local CUDA device
- [x] Bounded vanilla vLLM model-load smoke is captured for the target hybrid model
- [x] Bounded vanilla vLLM one-prompt generation smoke is captured
- [x] Read-only Path C shim audit identifies vLLM 0.21.0 integration points
- [x] Observe-first vLLM cache-plan translator handles fake attention, Mamba, and hybrid configs without a vLLM dependency
- [x] Direct real vLLM cache planning dataclasses are translated into a Cachepawl observation artifact
- [x] Runtime-resolved vanilla vLLM `KVCacheConfig` is translated into a Cachepawl observation artifact
- [x] Reusable observe-first runtime vLLM cache-plan helper exists without a vLLM dependency
- [x] `ruff`, `ruff format --check`, `mypy`, and pytest status are recorded for the skeleton step

## Constraints

- Use `research/avmp/v2/VLLM_INTEGRATION_ROADMAP.md` as the controlling plan.
- Start with Path C: subclass `KVCacheManager` and inject via scheduler configuration.
- Pin `vllm==0.21.0` for reproducibility.
- Use Zamba2-2.7B-instruct as primary and Falcon-H1-1.5B-Instruct as backup.
- Keep `TritonAVMPAllocator` as correctness-oracle context only; production per-allocate Triton deployment remains deferred to v2.1.

## Non-Goals

- Do not refactor unrelated allocator or benchmark code.
- Do not change paper claims without updating the relevant research notes.
- Do not bypass failed checks by weakening tests, narrowing behavior, or deleting validation.

## Risks

- vLLM internals may not subclass cleanly; use the documented Path A fork fallback if Path C fails.
- WSL2 or 12 GiB GPU memory pressure may force the Falcon-H1 backup model.
- If vLLM upstream fixes the hybrid cache overestimation before evaluation, update the paper narrative before claiming results.

## Progress Notes

- 2026-05-23: Landed import-safe `cachepawl.integrations.vllm` skeleton and
  structural cache-plan translation records. Runtime vLLM serving, allocator
  replacement, and baseline metrics remain pending.
- 2026-05-23: Landed planner-only baseline measurement spine for deterministic
  synthetic cache-probe JSONL records. Runtime vLLM baseline capture remains
  pending.
- 2026-05-23: Landed planner-comparison path for vLLM-style padded modeling
  baseline versus Cachepawl AVMP planning. The comparison is CPU-safe and does
  not require vLLM.
- 2026-05-23: Corrected planner benchmark metric semantics before committing
  artifacts by replacing ambiguous `waste_ratio` wording with explicit
  `overestimation_ratio` and `wasted_fraction` fields.
- 2026-05-23: Generated the first committed-reference planner-comparison
  artifact pack at `benchmarks/results/rtx3060/planner-comparison/`.
- 2026-05-23: Added pinned vanilla vLLM baseline capture path and recorded a
  structured not-runnable result because vLLM is not installed, CUDA is
  unavailable from torch, and `nvidia-smi` cannot initialize NVML.
- 2026-05-23: Accepted D004 to fix local WSL2 GPU/NVML access before creating
  the pinned vLLM environment or moving the runtime baseline to another GPU
  host.
- 2026-05-23: Updated D004 after local WSL2 GPU/NVML visibility was restored.
  The runtime baseline blocker artifact now records CUDA available on the RTX
  3060, with missing vLLM as the remaining blocker.
- 2026-05-23: Created the isolated `/tmp/vllm-cachepawl-venv`, installed
  `vllm==0.21.0` there, validated CUDA visibility inside it, and updated the
  baseline artifact to `ready` using `PYTHONPATH=src` without editable install.
- 2026-05-23: Added bounded runtime smoke capture and loaded
  `Zyphra/Zamba2-2.7B-instruct` with vanilla `vllm==0.21.0` on the local RTX
  3060. The baseline artifact is now `completed` for model-load smoke only;
  serving, generation, allocator replacement, and shim behavior remain pending.
- 2026-05-23: Added bounded one-prompt generation capture and recorded
  successful vanilla vLLM generation for `Zyphra/Zamba2-2.7B-instruct` with 13
  prompt tokens, 8 generated tokens, and no long-lived serving or shim behavior.
- 2026-05-23: Completed the read-only Path C shim audit against the installed
  `vllm==0.21.0` package. Accepted D005 to observe and translate resolved vLLM
  cache plans before mutating scheduler or allocator behavior.
- 2026-05-25: Added the observe-first vLLM cache-plan translator for
  duck-typed `AttentionSpec`, `MambaSpec`, `KVCacheGroupSpec`,
  `KVCacheTensor`, and `KVCacheConfig`-like objects. The translator remains
  import-safe without vLLM installed and performs no runtime mutation.
- 2026-05-25: Recreated the isolated `/tmp/vllm-cachepawl-venv`, installed
  pinned `vllm==0.21.0`, and captured a direct real-object cache-plan
  translation artifact under
  `research/avmp/v2/results/vllm-cache-plan-observation/`. This did not load a
  model, call `get_kv_cache_configs`, modify vLLM, monkeypatch, replace
  allocators, or implement Path C mutation.
- 2026-05-25: Captured a bounded runtime-resolved vanilla vLLM
  `KVCacheConfig` from
  `LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config` for
  `Zyphra/Zamba2-2.7B-instruct`. The translated artifact records 329 blocks,
  7 cache groups, and 9 cache tensors with no vLLM mutation or long-lived
  serving.
- 2026-05-25: Promoted the bounded runtime cache-plan object walk into an
  import-safe `observe_vllm_runtime_cache_plan` helper. The capture script now
  uses the reusable helper while preserving artifact format and observe-only
  behavior.
