try:
    from .regex_tokenizer import RegexTokenizer
    from .patterns import INDIC_SPLIT_PATTERN
except ImportError:  # allow running as a script from inside `tokenizer/`
    from regex_tokenizer import RegexTokenizer
    from patterns import INDIC_SPLIT_PATTERN
import argparse
import os
import random

_HERE = os.path.dirname(__file__)


def _assign_special_token_ids(merge_vocab_size: int, specials: list[str]) -> dict[str, int]:
    """
    Assign special tokens after the mergeable vocab.

    merge_vocab_size is the size of the mergeable vocab (bytes + merges), i.e. 256 + num_merges.
    """
    if merge_vocab_size < 256:
        raise ValueError(f"merge_vocab_size must be >= 256, got {merge_vocab_size}")
    return {tok: merge_vocab_size + i for i, tok in enumerate(specials)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train mgpt2 RegexTokenizer and save .model/.vocab artifacts.")
    ap.add_argument(
        "--corpus",
        default=os.path.join(_HERE, "tok_corpus.txt"),
        help="Path to UTF-8 corpus file (one example per line). Default: tokenizer/tok_corpus.txt",
    )
    ap.add_argument(
        "--num_merges",
        type=int,
        default=50000,
        help="Number of BPE merges to learn (excludes bytes + special tokens). GPT-2 uses 50000.",
    )
    ap.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="DEPRECATED: use --num_merges. If set, num_merges = (vocab_size - num_special) - 256.",
    )
    ap.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Optional: only read first N lines from corpus (fast smoke tests).",
    )
    ap.add_argument(
        "--sample_lines",
        type=int,
        default=None,
        help="Optional: reservoir-sample N non-empty lines from the corpus (representative + fast).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="RNG seed used with --sample_lines (and shuffling).",
    )
    ap.add_argument(
        "--exclude_lines_file",
        default=None,
        help="Optional: path to a text file (one line per example) whose lines must be excluded from training.",
    )
    ap.add_argument(
        "--max_chars",
        type=int,
        default=None,
        help="Optional: stop reading after N characters (fast smoke tests).",
    )
    ap.add_argument(
        "--min_chunk_freq",
        type=int,
        default=1,
        help="Ignore regex chunks that occur fewer times than this (speed knob).",
    )
    ap.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Only keep the top-N most frequent regex chunk types (speed knob).",
    )
    ap.add_argument(
        "--special",
        action="append",
        default=["<|endoftext|>"],
        help="Special token to reserve at the end of the vocab. Can be passed multiple times.",
    )
    ap.add_argument(
        "--out_prefix",
        default=os.path.join(_HERE, "artifacts", "mgpt2"),
        help="Output prefix for artifacts (writes <prefix>.model and <prefix>.vocab).",
    )
    ap.add_argument("--verbose", action="store_true", help="Print merge logs during training.")
    args = ap.parse_args()

    specials: list[str] = []
    seen = set()
    for s in args.special:
        if s in seen:
            continue
        seen.add(s)
        specials.append(s)

    if args.vocab_size is not None:
        # Backward compatible mode
        merge_vocab_size = (args.vocab_size - len(specials))
        num_merges = merge_vocab_size - 256
        if num_merges < 0:
            raise SystemExit("vocab_size is too small once special tokens are reserved.")
    else:
        num_merges = args.num_merges
        merge_vocab_size = 256 + num_merges

    final_vocab_size = merge_vocab_size + len(specials)
    special_tokens = _assign_special_token_ids(merge_vocab_size, specials)

    parts: list[str] = []
    total_chars = 0

    exclude: set[str] = set()
    if args.exclude_lines_file:
        with open(args.exclude_lines_file, "r", encoding="utf-8") as f:
            for raw in f:
                s = raw.rstrip("\n")
                if s:
                    exclude.add(s)

    if args.sample_lines is not None and args.sample_lines > 0:
        rng = random.Random(args.seed)
        reservoir: list[str] = []
        seen = 0
        with open(args.corpus, "r", encoding="utf-8") as f:
            for raw in f:
                s = raw.rstrip("\n")
                if not s:
                    continue
                if s in exclude:
                    continue
                seen += 1
                if len(reservoir) < args.sample_lines:
                    reservoir.append(s)
                else:
                    j = rng.randrange(seen)
                    if j < args.sample_lines:
                        reservoir[j] = s
        rng.shuffle(reservoir)
        parts = reservoir
    else:
        with open(args.corpus, "r", encoding="utf-8") as f:
            kept = 0
            for i, line in enumerate(f):
                if args.max_lines is not None and kept >= args.max_lines:
                    break
                if not line:
                    continue
                s = line.rstrip("\n")
                if not s:
                    continue
                if s in exclude:
                    continue
                parts.append(s)
                kept += 1
                total_chars += len(parts[-1]) + 1
                if args.max_chars is not None and total_chars >= args.max_chars:
                    break

    corpus = "\n".join(parts)
    if not corpus:
        raise SystemExit("Empty corpus after applying --max_lines/--max_chars filters.")

    tokenizer = RegexTokenizer(regex=INDIC_SPLIT_PATTERN)
    tokenizer.train(
        corpus,
        merge_vocab_size,
        verbose=args.verbose,
        min_chunk_freq=args.min_chunk_freq,
        max_chunks=args.max_chunks,
    )
    tokenizer.register_special_tokens(special_tokens)

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tokenizer.save(args.out_prefix)

    # Convenience print for wiring into GPTConfig.vocab_size
    print(
        f"Saved: {args.out_prefix}.model / {args.out_prefix}.vocab | "
        f"num_merges={num_merges} | merge_vocab_size={merge_vocab_size} | final_vocab_size={final_vocab_size} | "
        f"num_special={len(specials)} | "
        f"max_id={max(list(tokenizer.vocab.keys()) + list(tokenizer.inverse_special_tokens.keys()))}"
    )


if __name__ == "__main__":
    main()