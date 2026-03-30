"""HellaSwag evaluation on a trained checkpoint.

Thin wrapper around hellaswag.py (which has all the data/rendering logic).
The in-training HellaSwag in train.py is a quick progress check; this script
is the post-hoc, reproducible, full-10042-example eval that writes proper JSON.

Usage:
    python -m eval.hellaswag_eval \\
        --checkpoint runs/<run>/model_XXXXX.pt \\
        --tokenizer-kind gpt2 \\
        --out runs/<run>/hellaswag_eval.json

Optionally compare the HF GPT-2 model as a contextual (non-controlled) reference:
    python -m eval.hellaswag_eval \\
        --checkpoint runs/<run>/model_XXXXX.pt \\
        --tokenizer-kind gpt2 \\
        --hf-reference gpt2 \\
        --out runs/<run>/hellaswag_eval.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types as _types
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hellaswag import get_most_likely_row, iterate_examples, render_example
from eval.lm_eval import load_model


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HellaSwagResult:
    checkpoint:      str
    step:            int
    tokenizer_kind:  str
    tokenizer_sha256: str | None
    num_correct:     int
    num_total:       int
    acc_norm:        float
    hf_reference:   dict | None   # None if not run; labelled non-controlled when present


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_enc(tokenizer_kind: str, tokenizer_model: str):
    if tokenizer_kind == "gpt2":
        import tiktoken
        _e = tiktoken.get_encoding("gpt2")
        return _types.SimpleNamespace(encode=_e.encode)
    from tokenizer.regex_tokenizer import RegexTokenizer
    _tok = RegexTokenizer()
    _tok.load(tokenizer_model)
    return _types.SimpleNamespace(
        encode=lambda t: _tok.encode(t, allowed_special=set())
    )


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_hellaswag(model, enc, device: str) -> tuple[int, int]:
    """Run the full HellaSwag val set. Returns (num_correct_norm, num_total)."""
    device_type  = "cuda" if device.startswith("cuda") else "cpu"
    num_correct  = 0
    num_total    = 0

    for example in tqdm(iterate_examples("val"), total=10042, desc="HellaSwag", unit="ex"):
        _, tokens, mask, label = render_example(example, enc=enc)
        tokens = tokens.to(device)
        mask   = mask.to(device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(tokens)

        pred_norm = get_most_likely_row(tokens, mask, logits)
        num_correct += int(pred_norm == label)
        num_total   += 1

    return num_correct, num_total


@torch.no_grad()
def run_hellaswag_hf(model_name: str, device: str) -> dict:
    """Run HellaSwag on a HuggingFace model as a contextual reference.
    Results MUST be labelled non-controlled in any report (different training data).
    """
    from transformers import GPT2LMHeadModel
    import tiktoken
    from torch.nn import functional as F

    hf_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    hf_model.eval()
    enc = tiktoken.get_encoding("gpt2")

    num_correct = num_total = 0
    for example in tqdm(iterate_examples("val"), total=10042,
                        desc=f"HellaSwag (HF {model_name})", unit="ex"):
        _, tokens, mask, label = render_example(example, enc=enc)
        tokens = tokens.to(device)
        mask   = mask.to(device)

        logits = hf_model(tokens).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        flat_losses  = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_tokens.view(-1), reduction="none"
        ).view(tokens.size(0), -1)
        shift_mask   = mask[..., 1:].contiguous().float()
        avg_loss     = (flat_losses * shift_mask).sum(1) / shift_mask.sum(1)
        num_correct += int(avg_loss.argmin().item() == label)
        num_total   += 1

    acc = num_correct / num_total
    return {
        "model":        model_name,
        "num_correct":  num_correct,
        "num_total":    num_total,
        "acc_norm":     acc,
        "controlled":   False,
        "note": "Contextual reference only — different training data (WebText, English-only).",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",      required=True)
    ap.add_argument("--tokenizer-kind",  default="gpt2", choices=["gpt2", "mgpt2"])
    ap.add_argument("--tokenizer-model", default="tokenizer/artifacts/mgpt2.model")
    ap.add_argument("--device",          default="cuda")
    ap.add_argument("--hf-reference",    default=None,
                    help="HF model name to run as contextual reference (e.g. 'gpt2'). "
                         "Results labelled non-controlled.")
    ap.add_argument("--out",             required=True)
    args = ap.parse_args()

    device = args.device
    model, ckpt = load_model(args.checkpoint, device)
    enc         = _load_enc(args.tokenizer_kind, args.tokenizer_model)

    tok_sha = (_sha256(args.tokenizer_model)
               if args.tokenizer_kind == "mgpt2" else None)

    num_correct, num_total = run_hellaswag(model, enc, device)
    acc_norm = num_correct / num_total
    print(f"  HellaSwag acc_norm = {num_correct}/{num_total} = {acc_norm:.4f}")

    hf_ref = None
    if args.hf_reference:
        hf_ref = run_hellaswag_hf(args.hf_reference, device)
        print(f"  HF {args.hf_reference} acc_norm = "
              f"{hf_ref['num_correct']}/{hf_ref['num_total']} = {hf_ref['acc_norm']:.4f}  "
              f"[contextual — non-controlled]")

    result = HellaSwagResult(
        checkpoint=       args.checkpoint,
        step=             int(ckpt.get("step", -1)),
        tokenizer_kind=   args.tokenizer_kind,
        tokenizer_sha256= tok_sha,
        num_correct=      num_correct,
        num_total=        num_total,
        acc_norm=         acc_norm,
        hf_reference=     hf_ref,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = asdict(result)
    out["git_hash"] = _git_hash()
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  wrote → {out_path}")


if __name__ == "__main__":
    main()
