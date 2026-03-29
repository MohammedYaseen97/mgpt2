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
    ap.add_argument("--checkpoint",     required=True)
    ap.add_argument("--eval-manifest",  required=True,
                    help="Path to data/eval/manifest.json (bucketed heldout sets)")
    ap.add_argument("--tokenizer-kind", default="gpt2", choices=["gpt2", "mgpt2"])
    ap.add_argument("--tokenizer-model",default="tokenizer/artifacts/mgpt2.model")
    ap.add_argument("--device",         default="cuda")
    ap.add_argument("--out",            required=True, help="Output JSON path.")
    args = ap.parse_args()

    # Load bucketed heldout files from manifest
    manifest = json.loads(Path(args.eval_manifest).read_text(encoding="utf-8"))
    buckets: dict[str, list[str]] = {}
    for bucket_name, info in manifest["buckets"].items():
        lines = [ln.rstrip("\n") for ln in
                 Path(info["file"]).read_text(encoding="utf-8").splitlines() if ln.strip()]
        buckets[bucket_name] = lines

    _ = buckets  # TODO: remove when implemented

    raise NotImplementedError("Implement LM eval (perplexity overall + by bucket) and write args.out JSON.")


if __name__ == "__main__":
    main()

