# vLLM v1 integration audit for AVMP

Purpose: map the vLLM v1 KV-cache surface so the AVMP allocator can be wired
in for the v2 paper's end-to-end demo. All citations are GitHub permalinks
to `vllm-project/vllm` `main` at audit time (2026-05-22; checks of `main`
HEAD against `master` HEAD redirected to `main`, so `main` is canonical).
Cross-referenced docs: `SLOWDOWN_ROOT_CAUSE.md`, `GRAPH_REPLAY_FEASIBILITY.md`,
`PAPER_SECTION_5_DRAFT.md` for context on why this is the v2 paper's core
performance path.

## §1 Allocator architecture surface

vLLM's v1 KV cache layer is three concentric classes plus per-kind managers:

### `KVCacheCoordinator` (`vllm/v1/core/kv_cache_coordinator.py`)

- **Abstract base** [`KVCacheCoordinator` at line 33](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_coordinator.py#L33).
- Concrete subclasses:
  - [`KVCacheCoordinatorNoPrefixCache` at line 378](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_coordinator.py#L378) — used when `enable_caching=False`.
  - [`UnitaryKVCacheCoordinator` at line 419](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_coordinator.py#L419) — used when there is exactly one KV cache group (homogeneous models).
  - [`HybridKVCacheCoordinator` at line 470](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_coordinator.py#L470) — **the v1 paper's named target**; used when `len(kv_cache_groups) > 1` (hybrid Mamba/attention models).
- Factory: [`get_kv_cache_coordinator()` at lines 603-658](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_coordinator.py#L603) selects the concrete subclass via `enable_caching` (line 616) and `len(kv_cache_config.kv_cache_groups)` (line 623). **No plugin registry; selection is hardcoded.**

Abstract methods on the base class that AVMP needs to honor (each line range cited): `__init__` (35-73), `get_num_blocks_to_allocate` (75-128), `allocate_new_computed_blocks` (130-151), `allocate_new_blocks` (153-177), `cache_blocks` (179-191), `free` (193-201), `get_num_common_prefix_blocks` (203-216), `remove_skipped_blocks` (218-233), `get_blocks` (235-242), `find_longest_cache_hit` (244-248, abstract), `new_step_starts` (250-253).

### `KVCacheManager` (`vllm/v1/core/kv_cache_manager.py`)

- [Class at line 240](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L240). Constructor [lines 241-259](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L241) takes `KVCacheConfig`, `max_model_len`, `hash_block_size`, several flags (`enable_caching`, `use_eagle`, `log_stats`, `enable_kv_cache_events`, `dcp_world_size`, `pcp_world_size`), and a `KVCacheMetricsCollector`.
- Composes the coordinator via `self.coordinator` and exposes its block pool at [line 261](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L261) (`self.block_pool = self.coordinator.block_pool`) and the per-kind managers at [line 269](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L269) (`self.coordinator.single_type_managers`).
- **Main allocation method**: [`allocate_slots()` at lines 317-423](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L317). Signature takes a `Request`, `num_new_tokens`, plus six optional kwargs (`num_new_computed_tokens`, `new_computed_blocks`, `num_lookahead_tokens`, `num_external_computed_tokens`, `delay_cache_blocks`, `num_encoder_tokens`, `full_sequence_must_fit`). Returns `KVCacheBlocks | None`.
- [`free()` at line 454](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L454) and [`get_num_common_prefix_blocks()` at line 492](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L492).

### `SingleTypeKVCacheManager` (`vllm/v1/core/single_type_kv_cache_manager.py`)

- [Abstract base at line 42](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py#L42).
- Concrete subclasses, one per layer kind:
  - [`FullAttentionManager` at line 697](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py#L697)
  - [`SlidingWindowManager` at line 747](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py#L747)
  - [`ChunkedLocalAttentionManager` at line 838](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py#L838)
  - [`MambaManager` at line 945](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py#L945) — **the SSM-state path that AVMP's SSM pool corresponds to**.
- Public methods on the base: `get_num_blocks_to_allocate` (217), `allocate_new_computed_blocks` (289), `allocate_new_blocks` (338), `take_new_block_ids` (361), `cache_blocks` (367), `free` (407), `get_num_common_prefix_blocks` (543, abstract), `find_longest_cache_hit` (549, abstract classmethod), `remove_skipped_blocks` (600), `get_num_skipped_tokens` (637), `new_step_starts` (647).

### `BlockPool` (`vllm/v1/core/block_pool.py`)

- [Class at line 253](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py#L253). Single implementation; no factory; no plugin point.

### Composition graph (concise)

```
Engine
└── KVCacheManager (allocate_slots/free are the user-facing methods)
    └── KVCacheCoordinator (Hybrid | Unitary | NoPrefixCache, factory-selected)
        ├── BlockPool (one, shared)
        └── single_type_managers: list[SingleTypeKVCacheManager]
            ├── FullAttentionManager  ← KV pool consumer (today: padded to uniform page size)
            └── MambaManager          ← SSM-state consumer (today: padded up to attention size, hence #37121)
```

### Issue #37121 — the 7× overestimation bug AVMP solves

[vllm-project/vllm#37121](https://github.com/vllm-project/vllm/issues/37121) — "[Performance]: KV cache ~7x memory overestimation for hybrid Mamba/attention models (Qwen3.5)". **Status: OPEN; no fix PR linked.**

- **Model**: `Qwen/Qwen3.5-4B-AWQ` (hybrid attention + Mamba).
- **Magnitude**: ~7× overestimation. Mamba blocks are ~1.1 MiB at bf16; the unified-page-size logic pads them up to ~32 KiB/block (attention's page width), yielding **13.7 % utilization** of allocated memory.
- **Locus**: `vllm/v1/core/kv_cache_utils.py` — functions `get_max_concurrency_for_kv_cache_config`, `_report_kv_cache_config`, `unify_kv_cache_spec_page_size`, `get_kv_cache_config_from_groups`. The bug is in the padding/unification step: "vLLM's KV cache profiler treats all layers uniformly, inflating both the reported token capacity and the actual memory allocation" — applying attention's O(n) scaling to Mamba's O(1) constant state.

This is **exactly what AVMP exists to fix**: per-pool native page sizing (RFC 0001 §1) with separate KV-page and SSM-block stores. The v2 paper §4 evaluation can frame the demo as "AVMP eliminates the #37121 overestimation". The bug being OPEN at audit time is a strong narrative anchor: the demo isn't competing with a maintainer-blessed fix, it's the only fix on the table.

## §2 Plugin / extension surface

vLLM's [`vllm/plugins/__init__.py`](https://github.com/vllm-project/vllm/blob/main/vllm/plugins/__init__.py) exposes four entry-point groups (lines 14, 17, 20, 25):

- `vllm.general_plugins`
- `vllm.io_processor_plugins`
- `vllm.platform_plugins`
- `vllm.stat_logger_plugins`

**No allocator, KV cache, or coordinator plugin point.** The [official plugin design doc](https://docs.vllm.ai/en/latest/design/plugin_system.html) confirms exactly these four groups.

The one available hook:

- **`scheduler_cls` config field** (in `vllm/config/scheduler.py` and `vllm/engine/arg_utils.py`) accepts a class object OR a `"module.ClassName"` string. Discussed in [issue #16479](https://github.com/vllm-project/vllm/issues/16479) and [discuss.vllm.ai/t/2157](https://discuss.vllm.ai/t/where-to-start-for-implementing-custom-memory-block-aware-scheduling-in-vllm/2157), where maintainers explicitly bless subclassing the block manager / KV cache manager via this route.

The implication for AVMP: there's no typed extension surface. The integration must subclass `KVCacheManager` (or `HybridKVCacheCoordinator`), inject the subclass at engine init via `scheduler_cls` or a top-level monkey-patch, and accept that constructor-signature changes in vLLM minor releases will break the shim.

## §3 Hybrid model fit (RTX 3060 12 GiB)

VRAM budget math: 12 GiB total − weights − ~1 GiB activations − overhead = cache headroom. With AVMP's 2× physical footprint (RFC 0002 §4.3, `BackingStore` pre-allocates at `total_bytes` for both pools), usable AVMP pool is **headroom ÷ 2**.

| Model | Params | bf16 weights | Cache headroom | AVMP pool (÷2) | vLLM v1 status | Verdict |
|---|---|---|---|---|---|---|
| **Zamba2-2.7B-instruct** (Zyphra) | ~3.0B | ~6 GiB | ~5 GiB | **~2.5 GiB** | ✓ [`vllm/model_executor/models/zamba2.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/zamba2.py) | **PRIMARY** — asymmetric arch (45 Mamba2 + 9 attention-augmented hybrid blocks) gives strong "uneven pool pressure" narrative for the paper |
| **Falcon-H1-1.5B-Instruct** (TII) | ~1.5B | ~3 GiB | ~8 GiB | **~4 GiB** | ✓ [`vllm/model_executor/models/falcon_h1.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/falcon_h1.py) | **BACKUP** — parallel attention+SSM each layer, more cache budget under 2× footprint, safer for Week-1 GO/NO-GO |
| Jamba 1.5 Mini | ~12B | ~24 GiB | does not fit | n/a | ✓ supported | too big |
| RecurrentGemma 2B | ~2B | ~4 GiB | n/a | n/a | ✗ Google never upstreamed Griffin/RG-LRU to vLLM v1 | unavailable |
| Bamba-9B, NemotronH-8B, Granite-4.0-Tiny | 7-9B | 14-18 GiB | does not fit | n/a | ✓ supported | too big |

Hybrid models are first-class in vLLM v1 as of 2025 ([PyTorch blog](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/), [v1 user guide](https://docs.vllm.ai/en/stable/usage/v1_guide/)). Known v1 limitation: no prefix caching for hybrids — irrelevant to AVMP demo (the demo focuses on allocation efficiency, not prefix reuse).

## §4 Integration-path bake-off

Three paths, evidence-driven evaluation:

### Path A — Fork vLLM

- **Files to fork-modify**: `vllm/v1/core/kv_cache_coordinator.py` (lines 470-602 = `HybridKVCacheCoordinator`), `vllm/v1/core/kv_cache_utils.py` (the padding bug locus), small wiring in `vllm/v1/core/kv_cache_manager.py`. ~300-500 LOC net.
- **Maintenance burden**: file-level git history shows `kv_cache_manager.py` ~10 commits since Dec 2025 (~biweekly), `kv_cache_utils.py` ~15 commits Jan-May 2026 (1-2 per week). v0.20.0 (2026-04-27) was hard-breaking (PyTorch 2.11, CUDA 13.0 default, Transformers v5, metrics rework). **Every rebase is a debug cycle.**
- **Workshop demo quality**: high IF a single tag is pinned. `git clone <fork> && pip install -e . && vllm serve ...` reproduces cleanly.
- **Engineering effort**: 2-3 weeks initial + recurring rebase tax.
- **Risk to 13-week deadline**: medium-high.

### Path B — Upstream PR to vLLM `main`

- **PR scope**: a clean PR replacing or augmenting `HybridKVCacheCoordinator` would be 300-500 LOC + tests. The padding-bug fix from issue #37121 alone might be 100-200 LOC.
- **PR review pace**: a 20-PR snapshot at audit time shows v1/core changes happen 1-2× per week with mixed external-contributor and maintainer authorship. Non-trivial allocator PRs draw multi-reviewer multi-week reviews; maintainers (`njhill`, `heheda12345`, `KuntaiDu`) are actively rewriting the same surface (recent Mamba refactor PR #41126 just landed).
- **Risk**: would PR into a moving target. Median merge time for non-trivial PRs is not directly computable from listing UI but the evidence suggests 4-8 weeks at best.
- **Engineering effort**: 4-8 weeks (including review iterations).
- **Risk to 13-week deadline**: **high** — almost certainly slips. The RFC discussion alone for a 200+ LOC allocator change can take a month.

### Path C — Plugin / shim (recommended)

- Subclass `KVCacheManager` (`kv_cache_manager.py:240`) — override the coordinator construction in `__init__` to inject AVMP-backed coordinator; route `allocate_slots()` (lines 317-423) through AVMP.
- Inject via `scheduler_cls` config or a top-level Python monkey-patch in the engine startup hook.
- Pin `vllm==0.21.0` (released 2026-05-15, the current stable at audit time) so internals don't shift mid-implementation.
- **Files we add to cachepawl** (no vLLM fork): `src/cachepawl/integrations/vllm.py` (~150 LOC) — `CachepawlKVCacheManager(KVCacheManager)` subclass + factory + feature flag.
- **Fragility**: vLLM v0.22 release between Week 2 and Week 4 could break the shim if `KVCacheManager.__init__` signature changes (it does: see the v0.20 hard break). Pinned `0.21.0` mitigates for the workshop.
- **Engineering effort**: 1.5-2 weeks initial + per-release patches.
- **Risk to 13-week deadline**: **low** — fits comfortably in the 4-week roadmap.

### Side-by-side

| Path | Effort | Demo quality (1-5) | Risk to 2026-08-22 deadline | Recommend? |
|---|---|---|---|---|
| A — Fork | 2-3 wk + rebase tax | 4 (pinned-tag reproducibility) | medium-high | escape hatch only |
| B — Upstream PR | 4-8 wk | 5 (canonical) | high — likely slips | post-workshop follow-up |
| **C — Shim** | **1.5-2 wk** | **3 (reproducible with pinned vllm version)** | **low** | **PRIMARY** |

## §5 Recommended integration path

**Path C: subclass `KVCacheManager` + `scheduler_cls` injection, pinned `vllm==0.21.0`.**

Justification:

- **13-week deadline**: only Path C fits cleanly inside the 4-week implementation budget the roadmap allocates. Path A's rebase tax compounds with the per-release breakage; Path B's review pipeline doesn't terminate before 2026-08-22.
- **Solo-founder bandwidth**: no team to absorb rebase work or shepherd a PR through review.
- **Paper narrative quality**: Path C frames as "demo wedge"; the paper §6 (Future Work) names Path B as the upstreaming path. This is honest and a typical NeurIPS workshop framing — many workshop papers ship demos, not merged-upstream features.
- **Post-workshop trajectory**: the Path C shim is self-contained and easy to lift into a Path B PR when the v0.21 → v0.22 transition is stable. No code lock-in.

Escape hatch: if a vLLM release between Weeks 1-4 breaks the shim AND a single-day workaround isn't obvious, fall back to **Path A pinned to `v0.21.0` tag** — the same `KVCacheManager` modifications but applied in a vLLM fork. Adds ~0.5-1 week to Week 2.

## §6 Recommended test model

**PRIMARY: Zyphra/Zamba2-2.7B-instruct.**

- Hybrid arch: 54 layers = 45 Mamba2 + 9 hybrid attention-augmented blocks at indices [6, 12, 18, 24, 30, 36, 42, 47, 51] (per the model's `config.json`). The asymmetric KV:SSM layer ratio is exactly what AVMP exists for: 9-attention + 45-SSM means the two pools have wildly different demand patterns. Demonstrates AVMP's "rebalance under uneven pool pressure" story cleanly.
- vLLM v1 first-class: [`vllm/model_executor/models/zamba2.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/zamba2.py).
- Fits 12 GiB RTX 3060 with ~5 GiB cache headroom; AVMP pool budget ~2.5 GiB under the 2× footprint. Tight but workable.

**BACKUP: tiiuae/Falcon-H1-1.5B-Instruct.**

- Smaller (~3 GiB bf16 weights) leaves ~8 GiB cache headroom → ~4 GiB usable AVMP pool. Much more comfortable budget for Week-1 GO/NO-GO and Week-3 stress tests.
- Parallel attention+SSM per layer exercises both pools every step (different stress shape from Zamba2's asymmetric interleaving) — could be a backup narrative if Zamba2's signal is hard to surface cleanly.
- Clean upgrade path to Falcon-H1-3B if 1.5B is too small to surface AVMP wins.
- vLLM v1 first-class: [`vllm/model_executor/models/falcon_h1.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/falcon_h1.py), explicit v1-aware ("In V1 all attention/ssm layers must have different index in prefix").

Week-1 swap trigger: if the Zamba2 + AVMP combination shows < 1 GiB usable cache budget on the RTX 3060 (under `--gpu-memory-utilization 0.9`), swap to Falcon-H1-1.5B and re-run the baseline. The Week-1 environment is identical between the two models; the swap is a config change, not a re-setup.

## Citation freshness

This audit cites `main` (canonical branch on `vllm-project/vllm`) as of 2026-05-22. Permalinks use the `main` ref rather than a pinned SHA so the audit reads cleanly today; per the integration roadmap, the implementation pins `vllm==0.21.0` (released 2026-05-15, `vllm/__init__.py` `__version__`) for byte-stable reproducibility. If a maintainer reads this audit weeks later and a referenced line number has drifted, the class names remain stable and grepping `class KVCacheCoordinator` / `class KVCacheManager` / `class HybridKVCacheCoordinator` finds the current location.

## Open questions for Week 1

These are the architectural unknowns the audit cannot answer from a code read alone; they need actual environment + small experiments:

1. **`HybridKVCacheCoordinator` subclassability**: do the abstract methods on `KVCacheCoordinator` have stable enough internal contracts that an AVMP-backed subclass can satisfy them without re-implementing the entire `BlockPool` lifecycle? Best probed by writing a 50-line `AvmpHybridCoordinator(HybridKVCacheCoordinator)` stub and running it through vLLM's existing hybrid-model unit tests.
2. **`scheduler_cls` injection completeness**: does `scheduler_cls` actually let us substitute the coordinator, or does it only swap the scheduler and leave coordinator construction in vLLM's core? Best probed by reading the engine startup code paths between `EngineArgs` parsing and `KVCacheManager.__init__`.
3. **Per-decode-step `allocate_slots` call shape**: does each decode step call `allocate_slots()` once (per-request) or many times (per-token, per-layer)? This dictates whether the per-call CPU overhead from PR #49's Triton characterization is relevant to the vLLM throughput claim.
