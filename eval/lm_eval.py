"""
TODO: Implement LM evaluation for fair baseline comparisons.

Required outputs:
- overall perplexity on heldout set
- bucketed perplexity (latin/devanagari/kannada/mixed) on the exact same heldout lines

Design constraints:
- Must be deterministic given seeds + fixed heldout file
- Must accept two models (baseline vs mgpt2) and produce comparable outputs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from eval.buckets import bucket
from model import GPT, GPTConfig


@dataclass
class Result:
    name: str
    ppl_overall: float
    ppl_by_bucket: dict[str, float]


def load_model(checkpoint_path: str, device: str) -> GPT:
    """
    TODO: load a checkpoint saved by your training loop (train.py) and return a GPT model.

    You must document:
    - which checkpoint key you load
    - how you set vocab_size / block_size
    """
    raise NotImplementedError


def compute_perplexity(model: GPT, token_batches: list[torch.Tensor], device: str) -> float:
    """
    TODO: compute perplexity on a list of token sequences.

    Hint: use cross-entropy loss over next-token prediction, then exponentiate.
    """
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--heldout_text", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    args = ap.parse_args()

    # TODO: tokenize heldout_text appropriately for the model/tokenizer under evaluation.
    # For fair comparisons, you will run this script once per model/tokenizer pair.
    heldout_lines = [ln.rstrip("\n") for ln in Path(args.heldout_text).read_text(encoding="utf-8").splitlines() if ln.strip()]
    buckets: dict[str, list[str]] = {}
    for s in heldout_lines:
        buckets.setdefault(bucket(s), []).append(s)

    _ = buckets  # TODO: remove when implemented

    raise NotImplementedError("Implement LM eval (perplexity overall + by bucket) and write args.out JSON.")


if __name__ == "__main__":
    main()

