"""One-shot downloader for the ShareGPT-Vicuna prompt distribution.

Fetches ``anon8231489123/ShareGPT_Vicuna_unfiltered`` via huggingface
hub, extracts the first human turn from each conversation, applies a
word-count proxy for token length (``len(words) * 1.3``), samples
``DEFAULT_SAMPLE_SIZE`` conversations, and writes
``research/avmp/data/sharegpt_prompts.json``.

Run once::

    uv run --with datasets python research/avmp/scripts/download_sharegpt.py

The output JSON is checked into the repo so sweeps are reproducible
without re-downloading. Token estimation is intentionally a coarse
proxy and is documented as such in the paper.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import cast

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
OUT_PATH: Path = REPO_ROOT / "research/avmp/data/sharegpt_prompts.json"

DEFAULT_SAMPLE_SIZE: int = 5000
DEFAULT_SEED: int = 20260520
DATASET_REPO: str = "anon8231489123/ShareGPT_Vicuna_unfiltered"
WORD_TO_TOKEN_RATIO: float = 1.3


def _word_count_to_tokens(text: str) -> int:
    words = text.split()
    return max(1, round(len(words) * WORD_TO_TOKEN_RATIO))


def _extract_first_human_prompt(conv: Mapping[str, object]) -> str | None:
    turns = conv.get("conversations")
    if not isinstance(turns, list):
        return None
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from")
        value = turn.get("value")
        if role == "human" and isinstance(value, str) and value.strip():
            return value
    return None


def download(sample_size: int, seed: int, out_path: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub package not installed. Run with:\n"
            "  uv run --with huggingface_hub python research/avmp/scripts/download_sharegpt.py"
        ) from exc

    target_filename = "ShareGPT_V3_unfiltered_cleaned_split.json"
    print(f"Fetching {DATASET_REPO}:{target_filename} from huggingface hub ...")
    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=target_filename,
        repo_type="dataset",
    )
    print(f"Reading {local_path} ...")
    convs_raw = json.loads(Path(local_path).read_text())
    if not isinstance(convs_raw, list):
        raise SystemExit(f"expected top-level JSON list in {local_path}")

    prompts: list[dict[str, object]] = []
    seen = 0
    for row in convs_raw:
        if not isinstance(row, dict):
            continue
        seen += 1
        text = _extract_first_human_prompt(cast(Mapping[str, object], row))
        if text is None:
            continue
        conv_id = row.get("id")
        prompts.append(
            {
                "conversation_id": str(conv_id) if conv_id is not None else f"row{seen}",
                "prompt_tokens": _word_count_to_tokens(text),
                "prompt_chars": len(text),
            }
        )

    if len(prompts) < sample_size:
        raise SystemExit(
            f"only {len(prompts)} valid prompts after scanning {seen} rows; "
            f"expected >= {sample_size}"
        )

    rng = random.Random(seed)
    rng.shuffle(prompts)
    sampled = prompts[:sample_size]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sampled, indent=2) + "\n")
    tokens = [p["prompt_tokens"] for p in sampled]
    tokens_int = [int(cast(int, t)) for t in tokens]
    tokens_int.sort()
    median = tokens_int[len(tokens_int) // 2]
    p95 = tokens_int[int(len(tokens_int) * 0.95)]
    print(f"wrote {out_path.relative_to(REPO_ROOT)}: {len(sampled)} prompts")
    print(f"  prompt_tokens median={median} p95={p95} max={tokens_int[-1]}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="download_sharegpt.py")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="number of prompts to retain after deduplicated shuffle",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for the prompt sample (default %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_PATH,
        help="output JSON path (default %(default)s)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    download(args.sample_size, args.seed, args.output)


if __name__ == "__main__":
    main()
