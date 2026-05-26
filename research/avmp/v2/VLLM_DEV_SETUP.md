# vLLM dev environment for AVMP integration (WSL2 + RTX 3060)

One-page operations doc for the Week-1 environment setup in
`VLLM_INTEGRATION_ROADMAP.md`. Companion to `VLLM_INTEGRATION_AUDIT.md`.

## Versions (pinned)

| Component | Pin | Source |
|---|---|---|
| vLLM | **`vllm==0.21.0`** (released 2026-05-15) | `pyproject.toml` on tag `v0.21.0` |
| Python | **3.10** | vLLM `pyproject.toml` allows `>=3.10,<3.15`; we pick 3.10 for parity with cachepawl CI matrix |
| PyTorch | **`torch==2.11.0`** (vLLM's hard pin) | vLLM 0.21 `pyproject.toml` build-system block |
| CUDA | **13.0** (provided by torch 2.11.0 cu130 wheel) | vLLM 0.21 release notes; default torch wheel for that release |
| Triton | comes via torch (vLLM does not pin separately) | n/a |
| AVMP / cachepawl | editable install from this repo | `pip install -e /home/nxank4/personal/cachepawl` in the vLLM venv |

Note: the cachepawl dev venv runs `torch 2.12.0+cu130` (per the Week 1 sanity
checks in `SLOWDOWN_ROOT_CAUSE.md`). The vLLM venv will sit at `torch 2.11.0`.
The two venvs are isolated; no conflict. AVMP itself does not require torch
2.12 — the kernel code is `torch>=2.4` (per cachepawl `pyproject.toml`), so
running AVMP inside a torch 2.11 venv works for the v2 paper's Python
allocator path. (The Triton-backed correctness oracle is a separate concern
and stays in the cachepawl venv where Triton 3.7.0 is installed.)

## Isolation: separate venv at `~/.cache/cachepawl/vllm-cachepawl-venv`

Do **NOT** install vLLM into the cachepawl venv. Two reasons:

1. The torch version pins conflict (2.12.0+cu130 vs 2.11.0) and `pip` will
   downgrade cachepawl's torch silently.
2. vLLM's wheel pulls in ~60 transitive dependencies (xformers,
   flash-attn, etc.) that we don't want polluting the cachepawl test
   matrix.

Setup commands (for the roadmap to run at Week 1 start):

```bash
# Create an isolated durable venv outside the cachepawl repo
uv venv --python 3.10 ~/.cache/cachepawl/vllm-cachepawl-venv
source ~/.cache/cachepawl/vllm-cachepawl-venv/bin/activate

# Install vLLM (pre-built wheel; no source build needed)
uv pip install "vllm==0.21.0" torch==2.11.0

# Install cachepawl in editable mode so the AVMP Python module is importable
uv pip install -e /home/nxank4/personal/cachepawl

# Sanity check
python -c "import vllm; print(vllm.__version__)"
python -c "from cachepawl.allocator.avmp import AsymmetricVirtualPool; print('cachepawl OK')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Pre-built wheels for `vllm==0.21.0` exist for CUDA 13.0 + Python 3.10/3.11/3.12.
**No source build needed.** (If a future task does require a source build,
plan ~30-60 min on the RTX 3060 dev box; not measured precisely in any
sample we collected.)

Do **NOT** add `vllm` to `pyproject.toml`. AVMP is the package; vLLM is the
runtime that consumes it. The dependency direction is intentional.

## Disk footprint

| Item | Size |
|---|---|
| vLLM wheel + transitive deps | ~4-6 GiB |
| Zamba2-2.7B-instruct weights (HF download) | ~6 GiB on disk (bf16 safetensors) |
| Falcon-H1-1.5B-Instruct backup weights | ~3 GiB on disk |
| HuggingFace transformers cache | ~0.5 GiB |
| **Total budget** | **~15 GiB** under `~/.cache/`, including the durable vLLM venv |

Check `df -h ~` before Week 1 starts; ensure ≥ 20 GiB free.

## WSL2 known issues (material risk for Week-1 GO/NO-GO)

The reference machine per [CLAUDE.md](../../CLAUDE.md) is WSL2 Ubuntu on
RTX 3060. vLLM is **not officially supported** on WSL2; several open
issues are directly in our path:

1. **[#41619](https://github.com/vllm-project/vllm/issues/41619)** —
   *Qwen3.6 hybrid Mamba models fail KV cache allocation on RTX PRO 6000
   Blackwell + WSL2 — 16 GiB invisible CUDA overhead.* This is the **same
   workload family** as our demo (hybrid Mamba KV allocation, WSL2,
   consumer-class card). The Week-1 GO/NO-GO test is essentially "does
   vanilla vLLM serve our chosen hybrid model on this WSL2 box without
   hitting this bug?"
2. **[#43381](https://github.com/vllm-project/vllm/issues/43381)** — UVA
   not available on WSL2. Affects unified-memory paths; mitigated by an
   in-flight PR ([#43348](https://github.com/vllm-project/vllm/issues/43348)).
3. **[#39149](https://github.com/vllm-project/vllm/issues/39149)** —
   Triton LLVM segfault on WSL2 RTX 4090 with vLLM 0.19.0. Unrelated to
   our model selection but a marker that WSL2 + vLLM + Triton has
   surface-area issues.

Mitigations the roadmap budgets:

- Try `--gpu-memory-utilization 0.7` and `--max-num-seqs 16` if the
  Week-1 GO/NO-GO hits CUDA-OOM despite < 12 GiB declared usage.
- Try Falcon-H1-1.5B as the smaller-footprint model.
- If WSL2 is truly the blocker, escalate to a non-WSL2 GPU for the
  final eval. Decision point listed in the roadmap; no spend committed
  yet.

## Quick-reference: every Week-1 command

```bash
# 1. Pre-check the host environment
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader      # expect "NVIDIA GeForce RTX 3060, ~10800 MiB"
df -h /tmp ~                                                       # expect ≥ 20 GiB free in each
uname -a                                                           # confirm WSL2 (look for "WSL2" in release)

# 2. Set up the isolated venv
uv venv --python 3.10 ~/.cache/cachepawl/vllm-cachepawl-venv
source ~/.cache/cachepawl/vllm-cachepawl-venv/bin/activate
uv pip install "vllm==0.21.0" torch==2.11.0
uv pip install -e /home/nxank4/personal/cachepawl

# 3. Download model weights (~6 GiB; ~10 min on broadband)
huggingface-cli download Zyphra/Zamba2-2.7B-instruct
# (backup, only if Week-1 GO/NO-GO swap triggers)
# huggingface-cli download tiiuae/Falcon-H1-1.5B-Instruct

# 4. GO/NO-GO: vanilla vLLM serve
vllm serve Zyphra/Zamba2-2.7B-instruct \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 32 \
    > /tmp/vllm-baseline-stdout.log 2>&1 &
# Sleep 60 s, then check the log for "Application startup complete" or any traceback.
```

Roadmap Week-1 §1 details the success/failure branches from that GO/NO-GO.

## Why this doc exists separately from the audit

The audit (`VLLM_INTEGRATION_AUDIT.md`) is the architectural reference; the
roadmap (`VLLM_INTEGRATION_ROADMAP.md`) is the plan. This doc is the **ops
runbook** — copy-pasteable commands and the disk/network/GPU
prerequisites. Splitting them keeps the audit reusable and the roadmap
focused on milestones.
