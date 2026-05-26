# Work Log

## 2026-05-26 — T005 mutation-hook design gate completed

- Added `research/avmp/v2/PATH_C_MUTATION_HOOK_DESIGN_GATE.md`
- Compared pre-call wrapper, post-call advisory/diff only, controlled isolated
  return-value substitution, and Scheduler/EngineCore hook paths
- Accepted D009 selecting planner-stage post-call advisory/diff as the safest
  next bounded experiment before any mutation
- Required any future mutation to use a default-off feature flag, explicit
  opt-in, no mutation in normal CLI/advisory mode, structured before/after
  artifact, rollback path, and vanilla vLLM parity check
- Completed T005 and Sprint 3 without vLLM source edits, monkeypatching,
  allocator replacement, returned Cachepawl plans, scheduler or worker layout
  mutation, vLLM main-environment dependency changes, Triton kernels, copy
  kernels, LSDR, serving changes, or quality evaluation
- Verified `git diff --check` and PawlKit `view`/`check`; sandboxed PawlKit
  runs failed with npm DNS `EAI_AGAIN`, and approved network reruns passed with
  0 warnings

## 2026-05-26 — Sprint 3 and T005 opened

- Kept Sprint 2 and T002 completed after successful direct planner-stage replay
- Opened Sprint 3 for the next bounded Path C mutation-hook design gate
- Opened T005 to compare pre-call wrapper, post-call advisory/diff only,
  controlled isolated return-value substitution, and scheduler/EngineCore hook
  options
- Scoped T005 to design documentation only: required control point,
  correctness risk, rollback strategy, advisory-mode unchanged-output
  verification, later substitution changed-behavior verification, and required
  pre-mutation tests
- Kept vLLM source edits, monkeypatching, allocator replacement, returned
  Cachepawl plans, scheduler or worker layout mutation, Triton kernels, copy
  kernels, LSDR, serving changes, and quality evaluation out of scope
- Validated the opened tracker with `npx @codepawl/pawlkit@0.3.0 view` and
  `npx @codepawl/pawlkit@0.3.0 check`; initial sandboxed runs failed with npm
  DNS `EAI_AGAIN`, and approved network reruns passed with 0 warnings

## 2026-05-26 — T002 direct planner-stage replay completed

- Removed deep-copying of real vLLM runtime objects before planner-stage replay
  in `benchmarks/scripts/capture_vllm_planner_stage_observation.py`
- Added structured child payloads for inputs-reached replay failures, including
  `inputs_reached`, `replay_failed`, `deepcopy_failed`, and
  `get_kv_cache_configs_called`
- Added focused tests for deepcopy-failure classification, inputs-reached
  replay failure metadata, and successful direct replay artifact writing
- Reran the T002 observation through
  `~/.cache/cachepawl/vllm-cachepawl-venv`; it reached real `VllmConfig`,
  `KVCacheSpec` maps, and available-memory inputs, called
  `get_kv_cache_configs(...)`, and wrote
  `translated_planner_stage_config.json`
- The planner-stage artifact records `planner_stage_translation`, one worker,
  63 cache specs, available memory `2915421184`, planner output
  `num_blocks=329`, runtime scheduler `num_blocks=329`, no scheduler config
  change during replay, and `planner_matches_runtime_scheduler=true`
- Completed T002 and Sprint 2; no vLLM source edits, monkeypatching, allocator
  replacement, returned plans, scheduler mutation, worker layout mutation,
  vLLM main-environment dependency, Triton kernels, copy kernels, LSDR, serving
  changes, or quality evaluation were added

## 2026-05-26 — durable vLLM env rerun for T002

- Switched the primary pinned vLLM runtime path from `/tmp/vllm-cachepawl-venv` to `~/.cache/cachepawl/vllm-cachepawl-venv`
- Updated T002 runbook references and benchmark-script pinned-env metadata constants to prefer the durable path
- Verified the durable env with escalated GPU-visible execution: `vllm==0.21.0`, torch `2.11.0+cu130`, CUDA `13.0`, CUDA available, one device, `NVIDIA GeForce RTX 3060`
- Reran `benchmarks/scripts/capture_vllm_planner_stage_observation.py` through the durable env with `PYTHONPATH=src`
- The T002 artifact remains blocked, but the blocker changed: vLLM/FlashInfer sampling-op compilation references stale `/tmp/vllm-cachepawl-venv` source paths before `get_kv_cache_configs(...)` is reached
- No vLLM source edits, monkeypatching, allocator replacement, scheduler mutation, worker layout mutation, vLLM main-environment dependency, Triton kernels, copy kernels, LSDR, serving changes, or quality evaluation were added

## 2026-05-26 — diagnose-vllm documentation completed

- Added a compact README section for `cachepawl diagnose-vllm`
- Included the exact artifact-input command using `translated_runtime_cache_config.json`, optional `raw_safe_metadata.json`, and `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli`
- Documented generated `report.json`, `summary.md`, and `manifest.json`
- Documented that artifact-input mode requires no vLLM, GPU, or NVML; does not rerun vLLM, load a model, monkeypatch, replace allocators, or change vLLM behavior; and is advisory-only until a future mutation hook exists
- Recorded the current diagnostic result metrics in README
- Completed T004; T002 remains blocked by host GPU/NVML access

## 2026-05-26 — diagnose-vllm documentation task opened

- Kept T002 blocked by host GPU/NVML access until GPU/NVML visibility is restored
- Opened T004 for `cachepawl diagnose-vllm` CLI documentation and smoke examples
- Scoped T004 to README usage text, an example command using the existing runtime observation artifact, no-vLLM/no-GPU/no-NVML explanation, advisory-only safety text, and references to `report.json`, `summary.md`, and `manifest.json`
- Kept new runtime integration, T002 continuation, vLLM source edits, monkeypatching, allocator replacement, and vLLM dependency changes out of scope

## 2026-05-26 — diagnose-vllm CLI implemented

- Added a `cachepawl` console script entrypoint with `cachepawl diagnose-vllm`
- Added import-safe artifact-input diagnostics under `cachepawl.integrations.vllm.diagnose`
- Reused existing advisory and dry-run helpers to compute diagnostic metrics from translated runtime cache artifacts
- Added focused no-vLLM CLI tests for success, missing translated config file, invalid JSON, unsupported schema, and deterministic output
- Generated `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/` with `report.json`, `summary.md`, and `manifest.json`
- The generated report records `planner_advisory_available`, 329 blocks, 7 cache groups, 9 cache tensors, 63 layers, 2,910,781,440 observed reserved bytes, 1,679,258,112 observed useful/recommended bytes, 1,231,523,328 advisory savings bytes, 1.733373 overestimation ratio, and 0.423090 wasted fraction
- The CLI does not import vLLM, rerun vLLM, load a model, call `nvidia-smi`, require `/tmp/vllm-cachepawl-venv`, require GPU/NVML, modify vLLM, monkeypatch, replace allocators, add kernels, or continue T002
- Completed T003; Sprint 2 remains in progress because T002 is still blocked by host GPU/NVML access

## 2026-05-26 — diagnose-vllm CLI task opened

- Marked T002 blocked by host GPU/NVML access while keeping it open for rerun after GPU visibility is restored
- Recorded that the T002 blocker is not evidence that `get_kv_cache_configs(...)` is unsafe or unavailable
- Added T003 for `cachepawl diagnose-vllm` CLI productization
- Scoped the first CLI mode to artifact input: read `translated_runtime_cache_config.json`, read optional `raw_safe_metadata.json`, and emit `report.json` plus `summary.md`
- Required T003 to reuse the existing translator, observer, advisory, dry-run, and runtime observation artifact format
- Kept vLLM source edits, monkeypatching, allocator replacement, GPU/NVML requirements, Triton kernels, copy kernels, LSDR, serving changes, and quality evaluation out of scope
- Preserved `.pawl/version`, `.pawl/policy.yaml`, and `.pawl/migration-report.md` as committed PawlKit migration metadata

## 2026-05-25 — PawlKit dogfooding policy

- Added D008 to record PawlKit feedback only when normal Cachepawl tracker work creates real friction
- Documented that Cachepawl work should not proactively test PawlKit beyond normal `view`, `check`, and tracker update usage
- Documented the required fields for any future PawlKit friction note: command used, expected behavior, actual behavior, impact, workaround, and suggested fix
- Cachepawl product code and research direction remain unchanged

## 2026-05-25 — vLLM planner-stage observation blocker

- Added `benchmarks/scripts/capture_vllm_planner_stage_observation.py`
- Added `tests/bench/test_vllm_planner_stage_observation.py`
- Generated `research/avmp/v2/results/vllm-planner-stage-observation/` with `README.md`, `manifest.json`, `blocker.json`, and `raw_safe_metadata.json`
- The pinned vLLM environment has `vllm==0.21.0`, but torch reports CUDA unavailable in this session, so the script did not initialize vanilla `LLM`, did not reach real runtime `VllmConfig`, `KVCacheSpec` dictionaries, or available-memory inputs, and did not call `get_kv_cache_configs(...)`
- The blocker records safe static metadata for `vllm.v1.core.kv_cache_utils.get_kv_cache_configs(...)` and its `EngineCore._initialize_kv_caches` call site
- Stabilized the blocker diagnosis: the pinned vLLM env reports torch `2.11.0+cu130`, CUDA `13.0`, unavailable, 0 devices; the main uv env reports torch `2.12.0+cu130`, CUDA `13.0`, unavailable, 0 devices; `nvidia-smi` fails because GPU access is blocked by the operating system
- Inspected `.pawl/version`, `.pawl/policy.yaml`, and `.pawl/migration-report.md`; these are expected schema metadata generated by the current pinned PawlKit run and should be committed with the tracker update
- vLLM source edits, monkeypatching, returned plans, allocator replacement, scheduler or worker mutation, Triton kernels, copy kernels, LSDR, long-lived serving, and quality evaluation remain out of scope

## 2026-05-25 — Sprint 1 closeout and planner-stage observation task

- Closed Sprint 1 / T001 as completed for the observe-first vLLM integration boundary
- Recorded T001 outputs: planner benchmark spine and RTX 3060 target-profile artifact, pinned vLLM 0.21.0 baseline on WSL2 RTX 3060, bounded model-load and generation smoke, real vLLM dataclass translator, runtime-resolved `KVCacheConfig` observer, advisory diagnostic, and non-mutating planner dry-run probe
- Opened Sprint 2 / T002 for real planner-stage observation around `get_kv_cache_configs(...)`
- T002 remains observe-first and bounded: it may observe or safely call real planner-stage objects, but must produce a structured blocker instead of widening scope if private/runtime-only API access is unsafe
- vLLM source edits, monkeypatching, returned plans, allocator replacement, scheduler or worker mutation, Triton kernels, copy kernels, LSDR, long-lived serving, and quality evaluation remain out of scope

## 2026-05-23 — PawlKit setup for cachepawl

- Created `.pawl/` project operating folder
- Added active sprint tracker and task tracker
- Added decision log
- Added cachepawl-specific context files
- Added root `AGENTS.md` and Claude command/skill integration
- Completed Sprint 0 / T000 and opened Sprint 1 / T001 for vLLM integration
- Noted that `.agents/` and `.codex/` are read-only mounts in this workspace, so their optional adapter files could not be written here
- Verified PawlKit with `check` and `view`
- Verified repo checks: ruff, format check, mypy, pytest, and build
- No product code implemented

## 2026-05-23 — vLLM integration skeleton

- Added import-safe `cachepawl.integrations.vllm` planning skeleton
- Added frozen/slots cache-plan dataclasses and optional vLLM availability helpers
- Added focused tests under `tests/integration/vllm/`
- Updated README status wording to reflect implemented allocator prototypes and benchmark harnesses
- Verified: PawlKit check, ruff, format check, mypy, full pytest, and build
- Runtime vLLM serving, allocator replacement, monkeypatching, and Triton deployment remain out of scope

## 2026-05-23 — baseline measurement spine

- Added `cachepawl.bench` planner-probe package with JSONL result schema
- Added CPU-safe runtime/GPU environment capture without requiring CUDA or vLLM
- Added deterministic `short-heavy`, `long-heavy`, and `mixed` synthetic workloads
- Added RTX 3060 12GB benchmark config artifact
- Added `benchmarks/scripts/run_cache_probe.py` JSONL probe CLI
- Added tests for schema validation, serialization, environment fallbacks, deterministic workloads, and CLI output
- Verified: PawlKit check, ruff, format check, mypy, full pytest, and build
- Runtime vLLM serving, allocator replacement, monkeypatching, Triton kernels, and LSDR remain out of scope

## 2026-05-23 — planner comparison evidence

- Added vLLM-style padded cache-planning modeling baseline
- Added Cachepawl AVMP planner comparison path using the same synthetic workloads
- Added `benchmarks/scripts/compare_cache_planners.py` for JSONL records and Markdown/CSV summaries
- Added tests for planner behavior, schema round-trips, summary rendering, and deterministic CLI output
- Verified: PawlKit check, example comparison command, bench tests, ruff, format check, mypy, full pytest, and build
- Runtime vLLM serving, monkeypatching, allocator replacement, Triton kernels, copy kernels, LSDR, and real inference remain out of scope

## 2026-05-23 — planner metric semantics correction

- Replaced ambiguous `waste_ratio` benchmark fields with explicit `overestimation_ratio` and `wasted_fraction` fields
- Defined `overestimation_ratio` as `estimated_bytes / useful_bytes`
- Defined `wasted_fraction` as `(estimated_bytes - useful_bytes) / estimated_bytes`
- Updated JSONL records, Markdown/CSV summaries, planner estimates, synthetic probe output, and tests to use the corrected names
- Runtime vLLM serving, monkeypatching, allocator replacement, Triton kernels, copy kernels, LSDR, and real inference remain out of scope

## 2026-05-23 — planner comparison artifact pack

- Added a deterministic artifact-pack generator for the planner comparison
- Generated `benchmarks/results/rtx3060/planner-comparison/` with per-workload JSONL, combined summary, environment metadata, manifest, and reproduction README
- Added a narrow `.gitignore` allowlist for the RTX 3060 planner-comparison reference pack
- Added tests for deterministic artifact-pack generation
- Runtime vLLM serving, monkeypatching, allocator replacement, Triton kernels, copy kernels, LSDR, and real inference remain out of scope

## 2026-05-23 — vanilla vLLM runtime baseline capture

- Added a pinned vLLM baseline capture runbook for the measurement-only runtime step
- Added `benchmarks/scripts/capture_vllm_baseline.py` to record vLLM/CUDA/GPU availability and runtime blockers without requiring vLLM
- Captured `research/avmp/v2/results/vllm-baseline/` with a structured `not_runnable` JSONL record and manifest
- Recorded that vLLM is not installed in the active uv environment, torch reports CUDA unavailable, and `nvidia-smi` cannot initialize NVML
- Runtime vLLM serving, monkeypatching, allocator replacement, Triton kernels, copy kernels, LSDR, and real inference remain out of scope

## 2026-05-23 — vLLM runtime baseline infrastructure decision

- Added D004 choosing local WSL2 GPU/NVML repair before creating the pinned vLLM environment
- Updated the runtime baseline blocker artifact with the full blocker chain and selected infrastructure path
- Deferred separate Linux GPU and rented cloud GPU paths unless local WSL2 GPU access cannot be restored
- Runtime vLLM serving, monkeypatching, allocator replacement, Path C shim work, Triton kernels, copy kernels, LSDR, and real inference remain out of scope

## 2026-05-23 — WSL2 GPU visibility restored for vLLM baseline

- Updated D004 to record restored local WSL2 GPU/NVML visibility
- Re-ran the vLLM baseline capture; the artifact now records CUDA available on the NVIDIA GeForce RTX 3060 with one torch CUDA device
- Updated the remaining blocker to missing vLLM in the active Cachepawl environment
- Did not create the pinned vLLM environment or add vLLM to the main environment

## 2026-05-23 — isolated pinned vLLM environment

- Created `/tmp/vllm-cachepawl-venv` with Python 3.10
- Installed pinned `vllm==0.21.0` inside the isolated environment with `uv pip install "vllm==0.21.0" --torch-backend=auto`
- Validated vLLM import and CUDA visibility inside the pinned env on the local RTX 3060
- Re-ran `capture_vllm_baseline.py` through the pinned env with `PYTHONPATH=src`; editable Cachepawl install was not needed
- Updated `research/avmp/v2/results/vllm-baseline/` to `status=ready`
- Runtime serving, model loading, monkeypatching, allocator replacement, Path C shim work, Triton kernels, copy kernels, LSDR, and model quality evaluation remain out of scope

## 2026-05-23 — bounded vanilla vLLM model-load smoke

- Added a bounded runtime smoke mode to `benchmarks/scripts/capture_vllm_baseline.py`
- Ran the capture through `/tmp/vllm-cachepawl-venv` with `PYTHONPATH=src`; editable Cachepawl install was still not needed
- Loaded `Zyphra/Zamba2-2.7B-instruct` with vanilla `vllm==0.21.0` on the local RTX 3060 using a 1200 second timeout
- Updated `research/avmp/v2/results/vllm-baseline/` to `status=completed` for bounded model-load smoke
- Recorded vLLM observations: 5.07 GiB model memory, 2.12 GiB available KV cache memory, 11,442 GPU KV cache tokens, 2.79x max concurrency for 4,096-token requests, and 43.12% Mamba page-size padding
- Long-lived serving, generation, model-quality evaluation, monkeypatching, allocator replacement, Path C shim work, Triton kernels, copy kernels, and LSDR remain out of scope

## 2026-05-23 — bounded vanilla vLLM generation smoke

- Added one-prompt bounded generation capture to `benchmarks/scripts/capture_vllm_baseline.py`
- Reused `/tmp/vllm-cachepawl-venv` with pinned `vllm==0.21.0` and `PYTHONPATH=src`; editable Cachepawl install was not needed
- Captured vanilla generation for `Zyphra/Zamba2-2.7B-instruct` with `max_model_len=4096`, `gpu_memory_utilization=0.7`, `max_num_seqs=1`, and `max_new_tokens=8`
- Preserved the existing model-load smoke evidence and added a second JSONL record for bounded generation
- Recorded prompt tokens 13, generated tokens 8, elapsed 43.399474 seconds, 0.184334 tokens/sec, and 10,905,399,296 bytes available GPU memory after generation
- Long-lived serving, model-quality evaluation, monkeypatching, allocator replacement, Path C shim work, Triton kernels, copy kernels, and LSDR remain out of scope

## 2026-05-23 — Path C shim audit

- Added `research/avmp/v2/PATH_C_SHIM_AUDIT.md`
- Audited installed `vllm==0.21.0` cache planning, hybrid cache, Mamba cache, scheduler, manager, and GPU runner paths under `/tmp/vllm-cachepawl-venv`
- Identified `KVCacheSpec`, `MambaSpec`, `KVCacheConfig`, `get_kv_cache_configs`, `Scheduler.__init__`, `KVCacheManager`, `HybridKVCacheCoordinator`, `MambaManager`, and `GPUModelRunner.initialize_kv_cache` as the key integration surfaces
- Added D005 choosing observe-first translation of vLLM cache plans before any scheduler or allocator mutation
- vLLM source edits, monkeypatching, allocator replacement, Path C shim behavior, Triton kernels, copy kernels, LSDR, long-lived serving, and quality evaluation remain out of scope

## 2026-05-25 — observe-first vLLM cache-plan translator

- Added `cachepawl.integrations.vllm.translator` with import-safe, duck-typed translation for vLLM-like cache planning objects
- Added serializable Cachepawl-owned records for translated cache specs, groups, tensors, and full cache configs
- Added fake-object tests for attention specs, Mamba/state specs, hybrid cache configs, alias handling, deterministic serialization, and typed unsupported-object errors
- Updated the Path C audit limitations to record that this step observes cache plans only and performs no vLLM mutation
- vLLM source edits, monkeypatching, allocator replacement, Scheduler/KVCacheManager injection, Triton kernels, copy kernels, LSDR, serving changes, and quality evaluation remain out of scope

## 2026-05-25 — PawlKit validation path restore

- Restored the reproducible PawlKit command as `npx @codepawl/pawlkit@0.3.0`
- Documented the pinned command in `.pawl/README.md` and `.pawl/context/REPO_COMMANDS.md`
- Validated current manual `.pawl/` edits with `view` and `check`; `check` passed with 0 warnings
- Confirmed that unscoped `npx pawlkit` is the wrong package name for this repo's current tooling

## 2026-05-25 — vLLM cache-plan observation

- Recreated `/tmp/vllm-cachepawl-venv` and installed pinned `vllm==0.21.0` there only
- Added `benchmarks/scripts/capture_vllm_cache_plan_observation.py`
- Captured `research/avmp/v2/results/vllm-cache-plan-observation/` with direct real vLLM object translation
- Reached real vLLM `AttentionSpec`, `MambaSpec`, `KVCacheGroupSpec`, `KVCacheTensor`, and `KVCacheConfig` dataclasses
- Widened translator compatibility for real tuple-based `MambaSpec.shapes` and `MambaSpec.dtypes`
- Recorded that `get_kv_cache_configs(...)` was not called because it needs runtime `VllmConfig`, per-worker spec maps, and available-memory inputs
- Verified the main Cachepawl environment still has no vLLM dependency
- vLLM source edits, monkeypatching, allocator replacement, Scheduler/KVCacheManager injection, model loading, serving, Triton kernels, copy kernels, LSDR, and quality evaluation remain out of scope

## 2026-05-25 — runtime vLLM cache-plan observation

- Added `benchmarks/scripts/capture_vllm_runtime_cache_plan_observation.py`
- Captured `research/avmp/v2/results/vllm-runtime-cache-plan-observation/` with a bounded runtime-resolved vanilla vLLM cache-plan translation
- Reached `LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config` after offline `LLM` initialization for `Zyphra/Zamba2-2.7B-instruct`
- Translated the runtime `KVCacheConfig` with Cachepawl's import-safe translator; the artifact records 329 blocks, 7 cache groups, and 9 cache tensors
- Compared runtime-resolved fields against the direct dataclass observation and recorded that the same translator assumptions remain compatible
- Added a focused blocker-path test for environments without vLLM installed
- vLLM source edits, monkeypatching, allocator replacement, Path C mutation, long-lived serving, Triton kernels, copy kernels, LSDR, and quality evaluation remain out of scope

## 2026-05-25 — reusable runtime vLLM cache-plan observer

- Added `cachepawl.integrations.vllm.observer` with `observe_vllm_runtime_cache_plan(llm)`
- The helper safely walks the known vanilla runtime path `LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config`
- The helper translates the resolved runtime `KVCacheConfig` through the existing import-safe translator and returns deterministic serializable observation records
- Missing `llm_engine`, `engine_core`, `scheduler`, or `kv_cache_config` paths now produce a structured `unsupported` observation instead of an `AttributeError`
- Refactored `benchmarks/scripts/capture_vllm_runtime_cache_plan_observation.py` to call the reusable helper while preserving the runtime artifact schema
- Added fake-object observer tests under `tests/integration/vllm/test_observer.py`
- vLLM source edits, monkeypatching, allocator replacement, Path C mutation, long-lived serving, Triton kernels, copy kernels, LSDR, and quality evaluation remain out of scope

## 2026-05-25 — Path C decision gate

- Added `research/avmp/v2/PATH_C_DECISION_GATE.md`
- Added D006 choosing observer-in-the-loop advisory comparison before scheduler, manager, allocator, or worker allocation mutation
- Classified translated runtime fields as sufficient for planner-only comparison, observer-in-the-loop logging, and future planner recommendations
- Classified mutation control as still insufficient for replacing vLLM allocation, changing scheduler decisions, or changing tensor allocation layout
- The next implementation step is an import-safe advisory comparison helper that consumes observer output and emits vLLM-observed versus Cachepawl-recommended metrics
- vLLM source edits, monkeypatching, allocator replacement, Path C mutation, long-lived serving, Triton kernels, copy kernels, LSDR, and quality evaluation remain out of scope

## 2026-05-25 — vLLM runtime cache diagnostic

- Added `cachepawl.integrations.vllm.advisory` with `advise_vllm_runtime_cache_plan(...)`
- Added `benchmarks/scripts/create_vllm_cache_diagnostic.py`
- Generated `research/avmp/v2/results/vllm-runtime-cache-diagnostic/` with `report.json`, `summary.md`, and `manifest.json`
- The diagnostic classifies the runtime observation as `planner_advisory_available` with `observe_only` and `mutation_required_for_runtime_effect` flags
- The report records 329 blocks, 7 cache groups, 9 cache tensors, 63 layers, 2,910,781,440 observed reserved bytes, 1,679,258,112 advisory useful/recommended bytes, 1,231,523,328 advisory savings bytes, 1.733373 overestimation ratio, and 0.423090 wasted fraction
- Added focused fake/config tests for advisory metrics and diagnostic artifact generation
- vLLM source edits, monkeypatching, allocator replacement, scheduler/manager/worker mutation, Path C mutation, long-lived serving, Triton kernels, copy kernels, LSDR, and quality evaluation remain out of scope

## 2026-05-25 — Path C mutation-probe decision

- Added `research/avmp/v2/PATH_C_MUTATION_PROBE_DECISION.md`
- Added D007 selecting a planner-level dry-run probe as the next Path C mutation-probe design
- Compared planner-level, scheduler-construction, and worker allocation probe paths for control point, runtime effect, correctness risk, private API risk, RTX 3060 testability, code footprint, and fallback path
- Selected planner-level dry-run because it is the least invasive candidate that can test Cachepawl recommendation insertion before vLLM finalizes cache tensor sizes
- Deferred scheduler construction and worker allocation hooks due higher correctness/private API risk and poorer fit for the next smallest probe
- vLLM source edits, monkeypatching, allocator replacement, scheduler/manager/worker mutation, Path C mutation, long-lived serving, Triton kernels, copy kernels, LSDR, and quality evaluation remain out of scope

## 2026-05-25 — vLLM planner dry-run probe

- Added `cachepawl.integrations.vllm.dry_run` with `dry_run_vllm_planner_probe(...)`
- Added `benchmarks/scripts/create_vllm_planner_dry_run_probe.py`
- Generated `research/avmp/v2/results/vllm-planner-dry-run-probe/` with `README.md`, `manifest.json`, `dry_run_result.json`, and `summary.md`
- The dry-run artifact records `planner_dry_run_available`, `safe_for_advisory_only=true`, `returned_to_vllm=false`, and `vllm_behavior_changed=false`
- The dry-run computes 2,910,781,440 vanilla observed reserved bytes, 1,679,258,112 vanilla useful bytes, 1,679,258,112 Cachepawl proposed reserved bytes, 1,231,523,328 estimated savings bytes, 1.733373 overestimation ratio, and 0.423090 wasted fraction
- Added focused fake/config tests for dry-run metrics and artifact generation
- vLLM source edits, monkeypatching, allocator replacement, scheduler/manager/worker mutation, Path C mutation, long-lived serving, Triton kernels, copy kernels, LSDR, and quality evaluation remain out of scope
