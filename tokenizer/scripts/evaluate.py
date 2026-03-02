import argparse
import json
from dataclasses import dataclass
from typing import Iterable

import tiktoken

from tokenizer.regex_tokenizer import RegexTokenizer


@dataclass
class Metrics:
    name: str
    total_chars: int = 0
    total_bytes: int = 0
    total_tokens: int = 0
    tokens_per_line: list[int] | None = None
    bytes_per_line: list[int] | None = None

    def add(self, text: str, n_tokens: int) -> None:
        self.total_chars += len(text)
        b = len(text.encode("utf-8"))
        self.total_bytes += b
        self.total_tokens += int(n_tokens)
        if self.tokens_per_line is not None:
            self.tokens_per_line.append(int(n_tokens))
        if self.bytes_per_line is not None:
            self.bytes_per_line.append(int(b))

    def as_dict(self) -> dict:
        t = max(1, self.total_tokens)
        b = max(1, self.total_bytes)
        c = max(1, self.total_chars)
        out = {
            "name": self.name,
            "total_chars": self.total_chars,
            "total_bytes": self.total_bytes,
            "total_tokens": self.total_tokens,
            "tokens_per_1k_chars": self.total_tokens * 1000.0 / c,
            "tokens_per_1k_bytes": self.total_tokens * 1000.0 / b,
            "bytes_per_token": self.total_bytes / t,
            "chars_per_token": self.total_chars / t,
        }
        if self.tokens_per_line is not None:
            out.update(
                {
                    "p50_tokens_per_line": _quantile(self.tokens_per_line, 0.50),
                    "p95_tokens_per_line": _quantile(self.tokens_per_line, 0.95),
                }
            )
        if self.tokens_per_line is not None and self.bytes_per_line is not None:
            per_line = []
            for tok, by in zip(self.tokens_per_line, self.bytes_per_line):
                per_line.append(tok * 1000.0 / max(1, by))
            out["p95_tokens_per_1k_bytes_per_line"] = _quantile_float(per_line, 0.95)
        return out


def iter_lines(path: str, limit: int | None) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.rstrip("\n")
            if line:
                yield line


def _quantile(vals: list[int], q: float) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    idx = int(round(q * (len(s) - 1)))
    return int(s[idx])


def _quantile_float(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(round(q * (len(s) - 1)))
    return float(s[idx])


def _bucket(text: str) -> str:
    # Simple script-based bucket for reporting.
    # Devanagari: U+0900..U+097F, Kannada: U+0C80..U+0CFF
    has_deva = any("\u0900" <= ch <= "\u097f" for ch in text)
    has_knda = any("\u0c80" <= ch <= "\u0cff" for ch in text)
    has_latin = any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text)
    flags = (has_latin, has_deva, has_knda)
    if flags == (True, False, False):
        return "latin"
    if flags == (False, True, False):
        return "devanagari"
    if flags == (False, False, True):
        return "kannada"
    if flags == (False, False, False):
        return "other"
    return "mixed"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate tokenizer efficiency on a text file.")
    ap.add_argument("--text", required=True, help="Path to UTF-8 text file (one example per line).")
    ap.add_argument("--limit", type=int, default=2000, help="Max non-empty lines to evaluate.")
    ap.add_argument("--model", default=None, help="Path to a trained .model file for RegexTokenizer.")
    args = ap.parse_args()

    lines = list(iter_lines(args.text, args.limit))
    if not lines:
        raise SystemExit("No non-empty lines found.")

    # Pre-bucket the lines so all tokenizers see identical partitions
    buckets: dict[str, list[str]] = {}
    for s in lines:
        buckets.setdefault(_bucket(s), []).append(s)

    def eval_bucket(name: str, encode_fn, ls: list[str]) -> Metrics:
        m = Metrics(name=name, tokens_per_line=[], bytes_per_line=[])
        for s in ls:
            m.add(s, len(encode_fn(s)))
        return m

    out: dict = {"text": args.text, "limit": args.limit, "overall": [], "by_bucket": {}}

    # Baseline 1: tiktoken GPT-2 (monolingual reference — shows how far English-only BPE falls on Indic)
    enc_gpt2 = tiktoken.get_encoding("gpt2")
    out["overall"].append(
        eval_bucket("tiktoken_gpt2", lambda s: enc_gpt2.encode(s, allowed_special={"<|endoftext|>"}), lines).as_dict()
    )

    # Baseline 2: tiktoken cl100k (multilingual reference — the fair apples-to-apples comparison)
    enc_cl100k = tiktoken.get_encoding("cl100k_base")
    out["overall"].append(
        eval_bucket("tiktoken_cl100k_base", lambda s: enc_cl100k.encode(s, allowed_special="all"), lines).as_dict()
    )

    # Candidate: mgpt2 RegexTokenizer
    cand = RegexTokenizer()
    if args.model:
        cand.load(args.model)
    cand_name = "mgpt2_RegexTokenizer_candidate" + ("" if not args.model else f" ({args.model})")
    out["overall"].append(
        eval_bucket(cand_name, lambda s: cand.encode(s, allowed_special="all"), lines).as_dict()
    )
    if not args.model:
        out["note"] = "Candidate tokenizer is UNTRAINED (no --model supplied). Provide a trained .model for meaningful results."

    for bname, ls in sorted(buckets.items(), key=lambda kv: kv[0]):
        out["by_bucket"][bname] = [
            eval_bucket("tiktoken_gpt2", lambda s: enc_gpt2.encode(s, allowed_special={"<|endoftext|>"}), ls).as_dict(),
            eval_bucket("tiktoken_cl100k_base", lambda s: enc_cl100k.encode(s, allowed_special="all"), ls).as_dict(),
            eval_bucket(cand_name, lambda s: cand.encode(s, allowed_special="all"), ls).as_dict(),
        ]

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

