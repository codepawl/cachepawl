"""Synthetic workload generator for hybrid cache allocator benchmarks.

Four presets ship: ``uniform_short`` (short interactive turns),
``mixed_long`` (mixed short and long-context queries),
``agentic_burst`` (bursty Poisson arrivals with a heavy lognormal tail),
and ``sharegpt_replay`` (real ShareGPT prompt-length distribution
resampled from ``research/avmp/data/sharegpt_prompts.json``).
All presets use Jamba-1.5-Mini-class layer counts (4 attention plus 28
SSM); the cache-size formulas are dtype-aware and respect packed FP4 via
``cachepawl.quant.dtypes.bytes_per_element``.

RTX 3060 12GB budget. The preset ``num_requests`` defaults (512, 256,
256) and the Jamba-Mini layer profile are sized so that a non-pathological
allocator's peak working set stays under 10 GiB at BF16. Operators that
exceed those defaults are responsible for re-checking VRAM headroom.

References:
- Jamba layer profile: https://huggingface.co/docs/transformers/main/en/model_doc/jamba
- Mamba-2 layer profile: https://huggingface.co/state-spaces/mamba2-1.3b/blob/main/config.json
- KV cache bytes formula (factor of 2 for K and V, GQA uses num_kv_heads):
  https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- SSM per-sequence state bytes follow ``d_inner * d_state * dtype_bytes``
  as a best-effort approximation; the exact byte count varies across
  Mamba-2 reference implementations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from cachepawl.models.spec import AttentionLayerProfile, SSMLayerProfile
from cachepawl.quant.dtypes import DType, bytes_per_element

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Request:
    """One synthetic request in a workload trace.

    Times are virtual ticks where one tick equals one decode step. The
    runner walks events in tick order; arrival and departure times are
    integers so that ties are easy to break deterministically.
    """

    request_id: int
    prompt_len: int
    gen_len: int
    arrival_tick: int
    departure_tick: int


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """Declarative recipe for a benchmark workload.

    Identified by ``name``; the generator dispatches its distribution
    parameters off ``name``. Mutating the parameters of a preset means
    registering a new name.
    """

    name: str
    num_requests: int
    attention_layers: int
    ssm_layers: int
    attention_profile: AttentionLayerProfile
    ssm_profile: SSMLayerProfile
    dtype: DType
    seed: int


JAMBA_MINI_ATTN: Final[AttentionLayerProfile] = AttentionLayerProfile(
    num_kv_heads=8,
    head_dim=128,
)
JAMBA_MINI_SSM: Final[SSMLayerProfile] = SSMLayerProfile(
    d_inner=8192,
    d_state=16,
)

_JAMBA_MINI_ATTN_LAYERS: Final[int] = 4
_JAMBA_MINI_SSM_LAYERS: Final[int] = 28


def _make_preset(name: str, num_requests: int, seed: int) -> WorkloadSpec:
    return WorkloadSpec(
        name=name,
        num_requests=num_requests,
        attention_layers=_JAMBA_MINI_ATTN_LAYERS,
        ssm_layers=_JAMBA_MINI_SSM_LAYERS,
        attention_profile=JAMBA_MINI_ATTN,
        ssm_profile=JAMBA_MINI_SSM,
        dtype=DType.BF16,
        seed=seed,
    )


PRESETS: Final[dict[str, WorkloadSpec]] = {
    "uniform_short": _make_preset("uniform_short", num_requests=512, seed=1),
    "mixed_long": _make_preset("mixed_long", num_requests=256, seed=2),
    "agentic_burst": _make_preset("agentic_burst", num_requests=256, seed=3),
    "sharegpt_replay": _make_preset("sharegpt_replay", num_requests=512, seed=4),
}


# ShareGPT replay: resample prompt_tokens from a 5000-prompt sample of
# the public ShareGPT corpus. The JSON file is loaded lazily and cached
# at module scope so repeated generations of the same preset do not
# re-read it. Set ``SHAREGPT_PROMPTS_PATH`` to override the path
# (primarily for tests).
_SHAREGPT_DEFAULT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[4] / "research/avmp/data/sharegpt_prompts.json"
)
_SHAREGPT_PROMPT_TOKENS_CACHE: dict[str, IntArray] = {}


def _load_sharegpt_prompt_tokens() -> IntArray:
    """Load and cache the ShareGPT prompt_tokens array.

    The cache key is the absolute resolved path string, so a test that
    points ``SHAREGPT_PROMPTS_PATH`` at a different file gets a fresh
    array; production callers reuse the single default-path entry.
    """

    override = os.environ.get("SHAREGPT_PROMPTS_PATH")
    path = Path(override) if override else _SHAREGPT_DEFAULT_PATH
    key = str(path.resolve())
    cached = _SHAREGPT_PROMPT_TOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    if not path.is_file():
        raise FileNotFoundError(
            f"ShareGPT prompt sample not found at {path!s}; set SHAREGPT_PROMPTS_PATH "
            "to override or run research/avmp/scripts/download_sharegpt.py"
        )
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"ShareGPT prompt sample at {path!s} must be a non-empty JSON list")
    tokens: list[int] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"ShareGPT entry at index {idx} is not an object")
        value = entry.get("prompt_tokens")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"ShareGPT entry at index {idx} has non-int prompt_tokens {value!r}")
        tokens.append(value)
    arr: IntArray = np.asarray(tokens, dtype=np.int64)
    _SHAREGPT_PROMPT_TOKENS_CACHE[key] = arr
    return arr


def per_token_kv_bytes(profile: AttentionLayerProfile, dtype: DType) -> float:
    """Per-token KV bytes for one attention layer.

    Returns a float because packed dtypes such as FP4 carry 0.5 bytes per
    element. Factor of 2 covers K and V; GQA uses ``num_kv_heads``.
    """

    return 2.0 * profile.num_kv_heads * profile.head_dim * bytes_per_element(dtype)


def per_sequence_ssm_bytes(profile: SSMLayerProfile, dtype: DType) -> float:
    """Per-sequence SSM state bytes for one Mamba-2-style layer.

    Approximated as ``d_inner * d_state * bytes_per_element(dtype)``.
    The exact byte count varies across reference implementations; this
    formula is the published Jamba-Mini-class approximation.
    """

    return float(profile.d_inner) * float(profile.d_state) * bytes_per_element(dtype)


def generate_request_stream(spec: WorkloadSpec) -> list[Request]:
    """Generate the request trace for ``spec`` deterministically.

    Same ``spec.seed`` produces the same trace bit for bit on the same
    numpy version. The function never touches the global numpy random
    state.
    """

    rng = np.random.default_rng(spec.seed)
    if spec.name == "uniform_short":
        return _generate_uniform_short(spec, rng)
    if spec.name == "mixed_long":
        return _generate_mixed_long(spec, rng)
    if spec.name == "agentic_burst":
        return _generate_agentic_burst(spec, rng)
    if spec.name == "sharegpt_replay":
        return _generate_sharegpt_replay(spec, rng)
    raise ValueError(
        f"unknown workload preset name {spec.name!r}; registered presets are {sorted(PRESETS)}"
    )


def _generate_uniform_short(spec: WorkloadSpec, rng: np.random.Generator) -> list[Request]:
    prompt_lens = rng.integers(128, 1025, size=spec.num_requests, endpoint=False)
    gen_lens = rng.integers(64, 257, size=spec.num_requests, endpoint=False)
    return _assemble(spec, prompt_lens, gen_lens, arrival_ticks=_one_per_tick(spec.num_requests))


def _generate_mixed_long(spec: WorkloadSpec, rng: np.random.Generator) -> list[Request]:
    is_long = rng.random(size=spec.num_requests) < 0.2
    short_lens = rng.integers(512, 4097, size=spec.num_requests, endpoint=False)
    long_lens = rng.integers(16384, 65537, size=spec.num_requests, endpoint=False)
    prompt_lens = np.where(is_long, long_lens, short_lens)
    gen_lens = rng.integers(128, 513, size=spec.num_requests, endpoint=False)
    return _assemble(spec, prompt_lens, gen_lens, arrival_ticks=_one_per_tick(spec.num_requests))


def _generate_sharegpt_replay(spec: WorkloadSpec, rng: np.random.Generator) -> list[Request]:
    # Clamp prompt_tokens to [16, 4096] so a single 6708-token outlier
    # cannot trip pool-size limits; the unclamped ShareGPT distribution
    # has p95 around 810 tokens, so the clamp affects under 1% of draws.
    pool = _load_sharegpt_prompt_tokens()
    sampled = rng.choice(pool, size=spec.num_requests, replace=True)
    prompt_lens = np.clip(sampled.astype(np.int64), 16, 4096)
    # Output (generation) lengths: log-normal mean=4.5, sigma=1.0 maps
    # to a median of ~90 tokens and a long tail, matching ShareGPT
    # response statistics that run ~100 to 1000 tokens.
    raw_gen = rng.lognormal(mean=4.5, sigma=1.0, size=spec.num_requests)
    gen_lens = np.clip(raw_gen.astype(np.int64), 32, 2048)
    return _assemble(spec, prompt_lens, gen_lens, arrival_ticks=_one_per_tick(spec.num_requests))


def _generate_agentic_burst(spec: WorkloadSpec, rng: np.random.Generator) -> list[Request]:
    inter_arrival = rng.exponential(scale=1.0 / 10.0, size=spec.num_requests)
    cumulative = np.cumsum(inter_arrival)
    arrival_ticks = np.ceil(cumulative).astype(np.int64)
    raw_prompt = rng.lognormal(mean=8.5, sigma=1.0, size=spec.num_requests)
    prompt_lens = np.clip(raw_prompt.astype(np.int64), 64, 65536)
    gen_lens = rng.integers(64, 257, size=spec.num_requests, endpoint=False)
    return _assemble(spec, prompt_lens, gen_lens, arrival_ticks=arrival_ticks)


def _one_per_tick(num_requests: int) -> IntArray:
    return np.arange(num_requests, dtype=np.int64)


def _assemble(
    spec: WorkloadSpec,
    prompt_lens: IntArray,
    gen_lens: IntArray,
    arrival_ticks: IntArray,
) -> list[Request]:
    requests: list[Request] = []
    for idx in range(spec.num_requests):
        prompt_len = int(prompt_lens[idx])
        gen_len = int(gen_lens[idx])
        arrival = int(arrival_ticks[idx])
        requests.append(
            Request(
                request_id=idx,
                prompt_len=prompt_len,
                gen_len=gen_len,
                arrival_tick=arrival,
                departure_tick=arrival + prompt_len + gen_len,
            )
        )
    return requests
