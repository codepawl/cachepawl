"""Synthetic workload generator for hybrid cache allocator benchmarks.

Three presets ship: ``uniform_short`` (short interactive turns),
``mixed_long`` (mixed short and long-context queries), and
``agentic_burst`` (bursty Poisson arrivals with a heavy lognormal tail).
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

from dataclasses import dataclass
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
}


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
