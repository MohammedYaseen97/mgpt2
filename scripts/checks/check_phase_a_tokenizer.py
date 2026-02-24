"""
Phase A check: tokenizer artifacts exist and have expected ID ranges.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer.regex_tokenizer import RegexTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tokenizer/artifacts/mgpt2.model")
    ap.add_argument("--vocab_size", type=int, default=50257)
    args = ap.parse_args()

    p = Path(args.model)
    if not p.exists():
        raise SystemExit(f"Missing trained tokenizer model: {p}")

    tok = RegexTokenizer()
    tok.load(str(p))

    max_id = max(list(tok.vocab.keys()) + list(tok.inverse_special_tokens.keys()))
    if max_id != args.vocab_size - 1:
        raise SystemExit(f"Expected max_id={args.vocab_size-1}, got {max_id}")

    if "<|endoftext|>" not in tok.special_tokens:
        raise SystemExit("Missing required special token <|endoftext|>")

    print("OK: Phase A tokenizer check passed.")


if __name__ == "__main__":
    main()

