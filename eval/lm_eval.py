"""LM evaluation — perplexity overall + bucketed (latin/deva/knda/mixed).

RAM design (mirrors tokenize_shards.py rolling-buffer pattern):
  - Files are read lazily line-by-line; the full file is never loaded into RAM.
  - _iter_windows() maintains a rolling token buffer bounded to ~2×BLOCK_SIZE ints
    (~16 KB) regardless of file size.
  - Batches are processed immediately; no chunk list is ever accumulated.
  - Overall perplexity is derived from per-bucket (total_loss, total_tokens) sums,
    so each file is read exactly once.

Usage:
    python -m eval.lm_eval \\
        --checkpoint runs/<run>/model_XXXXX.pt \\
        --eval-manifest data/eval/manifest.json \\
        --tokenizer-kind gpt2 \\
        --out runs/<run>/lm_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import types as _types
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from model import GPT, GPTConfig

EOT_TOKEN  = 50256
BLOCK_SIZE = 1024
EVAL_BATCH = 8       # sequences per GPU call


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Result:
    checkpoint:     str
    step:           int
    tokenizer_kind: str
    max_lines:      int          # 0 = full heldout
    ppl_overall:    float
    ppl_by_bucket:  dict[str, float]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str) -> tuple[GPT, dict]:
    """Load a checkpoint written by train.py and return (eval-mode GPT, raw ckpt).

    Checkpoint keys (see train.py):
      "model"    — raw_model.state_dict()
      "config"   — raw_model.config  (a GPTConfig instance, not a dict)
      "step"     — int
      "val_loss" — float

    vocab_size / block_size come from the saved GPTConfig — correct for both
    the gpt2-tokenizer and mgpt2-tokenizer runs without any manual wiring.
    """
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device), ckpt


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def _load_enc(tokenizer_kind: str, tokenizer_model: str):
    """Return a SimpleNamespace with .encode(str) -> list[int]."""
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
# Rolling-buffer window generator  (same pattern as tokenize_shards.py)
# ---------------------------------------------------------------------------

def _iter_windows(lines: Iterable[str], enc) -> Iterator[list[int]]:
    """Stream (BLOCK_SIZE+1)-token windows from a line iterable.

    Uses a rolling buffer that stays bounded to ~2×BLOCK_SIZE ints (~16 KB)
    regardless of corpus size — same fill-and-flush pattern as tokenize_shards.py.
    Stride = BLOCK_SIZE so each token appears as a prediction target exactly once.
    """
    win = BLOCK_SIZE + 1
    buf: list[int] = []

    for line in lines:
        toks = enc.encode(line)
        if not toks:
            continue
        buf.append(EOT_TOKEN)
        buf.extend(toks)

        while len(buf) >= win:
            yield buf[:win]
            del buf[:BLOCK_SIZE]   # keep the last token (overlap with next window)


# ---------------------------------------------------------------------------
# Perplexity (streaming)
# ---------------------------------------------------------------------------

def compute_perplexity(
    model: GPT,
    lines: Iterable[str],
    enc,
    device: str,
    desc: str = "",
) -> tuple[float, float, int]:
    """Compute perplexity over a stream of text lines.

    Internally batches EVAL_BATCH windows at a time — at most
    EVAL_BATCH × (BLOCK_SIZE+1) × 8 bytes (~82 KB) of token data lives in RAM
    at any moment.

    Returns:
        (perplexity, total_weighted_ce_loss, total_tokens)

    The raw (total_loss, total_tokens) values let the caller derive overall
    perplexity across multiple buckets without re-reading any file:
        ppl_overall = exp(sum(losses) / sum(tokens))
    """
    device_type  = "cuda" if device.startswith("cuda") else "cpu"
    total_loss   = 0.0
    total_tokens = 0

    batch: list[list[int]] = []

    def _flush() -> None:
        nonlocal total_loss, total_tokens
        t = torch.tensor(batch, dtype=torch.long, device=device)  # (B, win)
        x = t[:, :-1]
        y = t[:, 1:].contiguous()   # .contiguous() required for model.py's .view()
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()
        batch.clear()

    for window in tqdm(_iter_windows(lines, enc), desc=desc or "perplexity", unit="win"):
        batch.append(window)
        if len(batch) == EVAL_BATCH:
            _flush()
    if batch:
        _flush()

    if total_tokens == 0:
        return float("inf"), 0.0, 0
    return math.exp(total_loss / total_tokens), total_loss, total_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _read_lines(path: str, max_lines: int) -> Iterator[str]:
    """Yield stripped lines from path, stopping after max_lines (0 = all)."""
    count = 0
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if ln.strip():
                yield ln
                count += 1
                if max_lines and count >= max_lines:
                    break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",      required=True)
    ap.add_argument("--eval-manifest",   required=True,
                    help="Path to data/eval/manifest.json (bucketed heldout sets)")
    ap.add_argument("--tokenizer-kind",  default="gpt2", choices=["gpt2", "mgpt2"])
    ap.add_argument("--tokenizer-model", default="tokenizer/artifacts/mgpt2.model")
    ap.add_argument("--device",          default="cuda")
    ap.add_argument("--max-lines",       type=int, default=2000,
                    help="Max lines per bucket (0 = full heldout)")
    ap.add_argument("--out",             required=True, help="Output JSON path.")
    args = ap.parse_args()

    device = args.device
    model, ckpt = load_model(args.checkpoint, device)
    enc         = _load_enc(args.tokenizer_kind, args.tokenizer_model)

    manifest = json.loads(Path(args.eval_manifest).read_text(encoding="utf-8"))

    ppl_by_bucket:  dict[str, float] = {}
    grand_loss   = 0.0
    grand_tokens = 0

    for bname, info in manifest["buckets"].items():
        lines = _read_lines(info["file"], args.max_lines)
        ppl, bl, bt = compute_perplexity(
            model, lines, enc, device, desc=f"  [{bname:6s}]"
        )
        ppl_by_bucket[bname] = ppl
        grand_loss   += bl
        grand_tokens += bt
        print(f"  ppl [{bname:6s}] = {ppl:>10.2f}")

    ppl_overall = math.exp(grand_loss / grand_tokens) if grand_tokens else float("inf")
    print(f"  ppl [overall] = {ppl_overall:>10.2f}")

    result = Result(
        checkpoint=     args.checkpoint,
        step=           int(ckpt.get("step", -1)),
        tokenizer_kind= args.tokenizer_kind,
        max_lines=      args.max_lines,
        ppl_overall=    ppl_overall,
        ppl_by_bucket=  ppl_by_bucket,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2))
    print(f"  wrote → {out_path}")


if __name__ == "__main__":
    main()
