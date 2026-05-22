# vLLM integration roadmap — 4 weeks to ML for Sys @ NeurIPS 2026

Companion docs: `VLLM_INTEGRATION_AUDIT.md` (architecture + path recommendation)
and `VLLM_DEV_SETUP.md` (env runbook). Workshop deadline:
**2026-08-22 (13 weeks from today, 2026-05-22)**. This roadmap budgets
4 weeks of implementation + 8 weeks of buffer for paper writing,
re-runs, and slip recovery.

**Integration path: Path C** — subclass `KVCacheManager` + inject via
`scheduler_cls` config, pinned `vllm==0.21.0`. Justification in
`VLLM_INTEGRATION_AUDIT.md §5`. Escape hatch: Path A (fork
`vllm-project/vllm` at tag `v0.21.0`) if Path C breaks on a vLLM release
between Weeks 1-4.

**Test model: Zamba2-2.7B-instruct (primary), Falcon-H1-1.5B-Instruct (backup)**.
Justification in `VLLM_INTEGRATION_AUDIT.md §6`. Week-1 swap trigger
defined below.

**Allocator under test: Python `AsymmetricVirtualPool`** — the v1
allocator. NOT `TritonAVMPAllocator`; per PR #51 the Triton variant is
the correctness oracle and its per-call Python overhead is the v2.1
story. The vLLM demo uses the Python AVMP to show end-to-end allocator
behavior; the Triton numbers stay in paper §5 as the hardware-realization
correctness claim.

## Week 1: env + baseline (2026-05-23 → 2026-05-30)

Goal: stand up the vLLM venv, reproduce the vanilla baseline, prove the
WSL2 + RTX 3060 + hybrid Mamba combination is workable.

**Steps**:

1. Run the `VLLM_DEV_SETUP.md` ops runbook in full (create venv, install
   `vllm==0.21.0`, install cachepawl editable, download Zamba2-2.7B
   weights).
2. **GO/NO-GO: vanilla vLLM serves Zamba2-2.7B on the RTX 3060.** Boot
   `vllm serve` with the parameters in the runbook. Pass = HTTP server
   accepts a small `/v1/completions` request and returns generated tokens
   without OOM. Fail branches:
   - **WSL2 / hybrid-Mamba bug (issue [#41619](https://github.com/vllm-project/vllm/issues/41619) family)**: try
     `--gpu-memory-utilization 0.7 --max-num-seqs 16`. If still crashes,
     swap to Falcon-H1-1.5B (runbook has the command).
   - **Falcon-H1 also fails on WSL2**: escalate to the user (Decision
     Point §4). Do not proceed to Week 2 until a working baseline exists.
3. **Reproduce the 7× overestimation** documented in vLLM issue [#37121](https://github.com/vllm-project/vllm/issues/37121).
   Confirm with a small ShareGPT-replay sample (≤ 64 prompts, max output
   256 tokens) that vanilla vLLM reports the inflated KV-block count for
   the chosen hybrid model. The bug is OPEN so this should land as
   expected; if it's somehow been fixed mid-flight, the paper narrative
   shifts (Decision Point §6 below).
4. **Measure the vanilla baseline** across ≥ 3 `--gpu-memory-utilization`
   values (e.g. 0.7, 0.8, 0.9). Per config, record:
   - `ttft_p50_ms` (time-to-first-token median across the sample)
   - `throughput_req_per_s`
   - `oom_count` (requests that failed with CUDA OOM)
   - `max_concurrent_requests` (server's reported peak)
   - `gpu_memory_utilization` config value
5. Commit the baseline numbers to
   `research/avmp/v2/results/vllm_baseline.json` plus a short
   `VLLM_BASELINE.md` summary (~5-10 lines) for paper §4 reuse.

**Deliverable (end of week)**: `vllm_baseline.json` exists; small
markdown summary exists; vanilla vLLM is known-runnable on this hardware.

**Week-1 swap trigger**: if Zamba2-2.7B + vanilla vLLM consistently
hits CUDA-OOM with `--gpu-memory-utilization 0.9` on this RTX 3060
*despite* under-12 GiB declared use, swap the chosen test model to
Falcon-H1-1.5B-Instruct, re-run steps 3-5, document the swap in the
roadmap closeout. Falcon-H1's 4 GiB AVMP pool budget is more
forgiving than Zamba2's 2.5 GiB under the 2× footprint.

## Week 2: AVMP shim (2026-05-30 → 2026-06-06)

Goal: wire the Python `AsymmetricVirtualPool` into vLLM via the Path C
shim. End the week with AVMP-enabled `vllm serve` running the same trace
as Week 1's baseline.

**Files to create in `cachepawl`**:

- `src/cachepawl/integrations/__init__.py`
- `src/cachepawl/integrations/vllm/__init__.py` — re-exports.
- `src/cachepawl/integrations/vllm/kv_cache_manager.py` —
  `CachepawlKVCacheManager(KVCacheManager)` subclass. Overrides
  `__init__` to construct an AVMP-backed coordinator instead of the
  default `HybridKVCacheCoordinator`. Forwards `allocate_slots` /
  `free` / `get_num_common_prefix_blocks` to the AVMP path.
- `src/cachepawl/integrations/vllm/coordinator.py` —
  `AvmpHybridCoordinator(HybridKVCacheCoordinator)` subclass that holds
  an `AsymmetricVirtualPool` instance and routes per-kind allocation
  through it.
- `src/cachepawl/integrations/vllm/__main__.py` — small entrypoint that
  reads `CACHEPAWL_AVMP_ENABLED` env var, builds the appropriate config,
  exec's `vllm serve` with `--scheduler-cls` pointing at our shim.

**Tests in `tests/integration/vllm/`** (new tree; CPU-only mocks where
possible):

- `test_shim_imports_cleanly.py` — assert `CachepawlKVCacheManager` is
  importable, is a subclass of `vllm.v1.core.kv_cache_manager.KVCacheManager`,
  has the expected constructor signature. Skips with informative message
  if `vllm` is not installed (CI does not install vLLM; this test only
  runs locally and in the vLLM venv).
- `test_avmp_coordinator_construct.py` — instantiate
  `AvmpHybridCoordinator` with a minimal `KVCacheConfig` derived from
  `kv_cache_groups = [attention_group, mamba_group]`, verify it builds
  the two AVMP pools at the expected sizes.

**End-of-week parity smoke**: AVMP-enabled `vllm serve` runs the same
small ShareGPT-replay sample as Week 1's baseline. Pass criteria:

- Server boots without crash with `CACHEPAWL_AVMP_ENABLED=1`.
- Per-request output tokens are byte-identical to the vanilla baseline
  (allocator change should not affect generation; the inference path
  reads from the same dtype'd KV memory, just with a different
  allocation policy).
- `oom_count` ≤ vanilla baseline.

**Deliverable**: `CachepawlKVCacheManager` shim ships, parity smoke
passes, code committed under `src/cachepawl/integrations/vllm/`.

**Fallback (if Path C breaks)**: if Week 2 discovers `KVCacheManager`
internals can't be cleanly overridden (constructor coupling to private
state, etc.), fall back to **Path A**: fork `vllm-project/vllm` at tag
`v0.21.0`, apply the same logical changes inline (`HybridKVCacheCoordinator`
and the `kv_cache_utils.py` padding bug locus), maintain as a separate
GitHub repo `codepawl/vllm-avmp-fork`. Adds 0.5-1 week to Week 2 budget;
recovery time absorbed by the 8-week post-Week-4 buffer.

## Week 3: end-to-end validation (2026-06-06 → 2026-06-13)

Goal: measure AVMP-vs-vanilla deltas at scale, surface any regressions,
produce paper §4 numbers.

**Steps**:

1. Re-run the Week-1 baseline trace with `CACHEPAWL_AVMP_ENABLED=1`. Record the
   same five metrics (`ttft_p50_ms`, `throughput_req_per_s`,
   `oom_count`, `max_concurrent_requests`, `gpu_memory_utilization`).
2. **Sweep**: at least 3 `--gpu-memory-utilization` settings (matching
   Week 1), and ideally 2-3 different `--max-num-seqs` settings to vary
   the request-concurrency pressure on AVMP's pool ratio.
3. **ShareGPT replay** if RTX 3060 budget allows. Sub-sample
   to ≤ 512 prompts to fit memory; reuse the prompt distribution from
   the v1 sweep `research/avmp/data/sharegpt_prompts.json`.
4. **Pairing**: every (workload, gpu_memory_utilization, max_num_seqs)
   tuple gets both a vanilla and an AVMP-enabled run; the JSON output
   records both for paired analysis.
5. **Bootstrap CI** on the throughput-ratio delta using
   `research/avmp/scripts/bootstrap_ci.py` (the v1 paper's protocol;
   B=10,000 paired resamples, seed=20260520).

**Pass criteria (Week-3 success)**:

- AVMP-enabled `oom_count` ≤ vanilla `oom_count` at every paired
  config (correctness — the allocator should never cause more OOMs
  than vanilla).
- AVMP-enabled `throughput_req_per_s` ≥ 0.95× vanilla AND ≥ 1.0× at
  the most memory-constrained config (the regime where AVMP's per-pool
  sizing should beat the unified-pool padding).
- The 7× overestimation from issue #37121 is empirically eliminated
  (AVMP-reported pool utilization > vanilla-reported utilization).

**Fail criteria → STOP and write findings**:

- AVMP-enabled `oom_count` > vanilla at multiple configs → AVMP has a
  regression in real vLLM context. Investigate, but if not resolved by
  end of Week 3, the paper §4 reverts to v1's simulator-only claims plus
  the Triton oracle from §5; the vLLM section becomes "future work."
- AVMP-enabled `throughput_req_per_s` < 0.95× vanilla universally → the
  per-call Python overhead documented in PR #49 dominates at scale.
  Same fallback as above.

**Deliverable**: `research/avmp/v2/results/vllm_avmp_comparison.json`
with paired metrics + bootstrap CI. Plus
`research/avmp/v2/VLLM_E2E_RESULTS.md` (~1 page) summarizing the
headline numbers for paper §4.

## Week 4: paper + submission (2026-06-13 → 2026-06-20)

Goal: convert v1 paper from acmart 11-page to NeurIPS workshop 4-page
extended abstract, integrate the vLLM end-to-end numbers, submit.

**Steps**:

1. Set up the NeurIPS 4-page LaTeX skeleton at `research/avmp/v2/sections/`.
   Use the workshop's official template (verify CFP at submission time).
2. **§1 Introduction**: condense v1's framing to 1-2 paragraphs; lead
   with issue #37121 as the motivating example.
3. **§2 Background**: existing vLLM HybridKVCacheCoordinator critique
   from v1 paper, ~0.5 page.
4. **§3 Method**: AVMP architecture diagram + the integration shim
   (Path C). Cite RFC 0001 + RFC 0002. ~1 page.
5. **§4 Evaluation**: vLLM end-to-end numbers from Week 3 + the
   simulator parity from PR #48. ~1.5 pages.
6. **§5 Hardware realization**: collapse `PAPER_SECTION_5_DRAFT.md`
   into a 0.25-page sidebar (correctness oracle + per-call overhead +
   deferred to v2.1). The full draft sits in the supplementary.
7. **§6 Future work**: Path B (upstream PR), v2.1 batched/deferred
   allocator API.
8. Anonymize per the workshop CFP's blinding policy (NeurIPS workshops
   are mixed: some single-blind, some double-blind; verify before
   submission).
9. Final review pass, generate PDF, commit under
   `research/avmp/v2/papers/v2-neurips-mlforsys-2026.pdf`, submit via
   the workshop's submission portal.

**Deliverable**: workshop submission complete; PDF committed.

## Risk register

| Risk | Likelihood | Trigger | Mitigation |
|---|---|---|---|
| Zamba2 + AVMP 2× footprint leaves < 1 GiB usable pool on 12 GiB | medium | Week 1 baseline GO/NO-GO sees memory-pressure-driven OOM | Week-1 swap to Falcon-H1-1.5B (runbook has the command) |
| WSL2 hits vLLM hybrid-Mamba bug ([#41619](https://github.com/vllm-project/vllm/issues/41619)) | medium | Week 1 baseline crashes despite < 12 GiB declared use | Try `--gpu-memory-utilization 0.7 --max-num-seqs 16`; then Falcon-H1 backup; then escalate to non-WSL2 GPU (Decision Point §4) |
| vLLM v0.22 release breaks Path C shim mid-roadmap | medium | `pip index versions vllm` shows v0.22 between Weeks 2-4 | Pin `vllm==0.21.0` in the venv; do NOT `pip install -U`. v0.22 changes absorb post-workshop. |
| `KVCacheManager` internals can't be cleanly subclassed | medium | Week 2 finds private-state coupling (e.g. tightly-bound `BlockPool` lifecycle) | Path A escape hatch: fork `vllm-project/vllm` at tag `v0.21.0`, apply same changes inline. +0.5-1 week, recoverable. |
| Issue #37121 gets fixed upstream before submission | low | vLLM PR closing #37121 lands between today and 2026-08-22 | Paper narrative shifts to "AVMP's per-pool sizing matches/exceeds the in-tree fix" with a comparison. Decision Point §6. |
| AVMP regression vs vanilla at scale (Week 3 fail criteria above) | low | Week 3 metrics fail | Fall back to simulator-only + Triton-oracle narrative; vLLM as future work. Mitigation already in fail-criteria branch above. |
| Workshop deadline slip | medium | Week 3 finishes with no clean win | Submit anyway with §3-§4 = v1 simulator + Triton oracle from §5; vLLM = "in-progress" |
| Recommended model removed from vLLM registry (rare) | low | Weights or model file disappears between Weeks 1-4 | Cache weights locally at Week 1; pin the `vllm==0.21.0` revision; document the model file's git SHA at audit time. |
| RTX 3060 hardware failure | low | machine doesn't boot / GPU not detected | escalate immediately; consider cloud GPU rental (~$0.50-1/hr for an L4) |
| `bootstrap_ci.py` reuse breaks if v1 protocol shifts | low | the existing script doesn't run | reimplement inline in the Week-3 analysis script (~20 LOC) |

## Decision points (user input needed before Week 1 starts)

These are blocking on the user's call; the roadmap has the
recommendation but the user is the final arbiter:

**1. Confirm or override the integration path.**
Recommended: **Path C** (`scheduler_cls` + `KVCacheManager` subclass,
pinned `vllm==0.21.0`). Audit §5 has the evidence.

**2. Confirm or override the primary test model.**
Recommended: **Zyphra/Zamba2-2.7B-instruct** primary, **tiiuae/Falcon-H1-1.5B-Instruct**
backup. Audit §6 has the evidence and the Week-1 swap trigger.

**3. vLLM PR draft early or wait?**
Recommended: **wait until post-workshop**. The 13-week budget doesn't
include a multi-week PR review pipeline, and the maintainers are
actively rewriting the same surface area (Mamba refactor PR #41126).
An early draft risks reviewer churn against unstable internals. Better
to ship the workshop demo on Path C, then submit a clean PR with
end-to-end data in hand.

**4. WSL2 vs cloud GPU fallback.**
Reference machine is WSL2 RTX 3060 12 GiB. If Week 1 GO/NO-GO fails
on WSL2 even with Falcon-H1 backup, do we (a) rent a small cloud GPU
for the workshop eval (L4 / T4 ~$0.50-1/hr, ~$50-100 total for the
Week-1 + Week-3 runs), or (b) change the paper scope?

**5. License decision for workshop submission.**
NeurIPS workshops are typically non-archival, so no license commitment
is required at submission. Confirm by the official CFP before submission.

**6. Issue #37121 fix-tracking.**
If a vLLM PR closing #37121 lands between today and 2026-08-22, what's
our framing? Current default: pivot the paper narrative to "AVMP's
per-pool sizing achieves or exceeds the in-tree fix on the same
workloads." User should confirm this is the desired stance vs e.g.
withdrawing the demo.

## What this roadmap does NOT do

- Does NOT install vLLM into the cachepawl venv (per task constraint
  and `VLLM_DEV_SETUP.md` rationale).
- Does NOT add vLLM as a dependency in cachepawl's `pyproject.toml`
  (per task constraint).
- Does NOT use the Triton-backed allocator for the vLLM demo. The
  Python `AsymmetricVirtualPool` is the integration target; the Triton
  variant stays as the correctness oracle per PR #51 closeout.
- Does NOT attempt Path B (upstream PR to vLLM `main`) within the
  4-week implementation budget. Documented as post-workshop follow-up.
- Does NOT commit Path C shim code; this is audit + roadmap only. Code
  ships in Week-2 PRs once the roadmap is approved.

## Cross-references

- `VLLM_INTEGRATION_AUDIT.md` — architecture + path recommendation + model fit
- `VLLM_DEV_SETUP.md` — Week-1 ops runbook
- `PAPER_SECTION_5_DRAFT.md` — what §5 Hardware Realization will say (now correctness oracle)
- `SLOWDOWN_ROOT_CAUSE.md` — PR #49: per-call CPU overhead
- `GRAPH_REPLAY_FEASIBILITY.md` — PR #50: why graph replay can't close the gap
- `TRITON_ROADMAP.md §Outcome and v2.1 deferral` — context on why the v2 perf story pivots to vLLM
