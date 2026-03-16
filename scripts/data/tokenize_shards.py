"""Tokenize corpus_mixture.txt into pretraining shards.

Reads the shuffled corpus, skips the eval slice (from data/eval/manifest.json),
splits the remainder into train (~98%) and val (~2%), tokenizes each document,
packs tokens into fixed-size shards, and writes them as int32 NumPy arrays.

Output layout (see data/README.md for full format spec):
    data/shards_{tokenizer}/
        train_000000.npy, train_000001.npy, …
        val_000000.npy
        manifest.json

Every document is prepended with the EOT token before packing.

Memory design:
    A single pre-allocated np.empty((shard_size,), dtype=np.int32) buffer is
    reused across all shards.  RAM cost is bounded to one shard buffer
    (~400 MB at 100 M tokens × 4 bytes) regardless of corpus size.
    Python lists of ints are never accumulated.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm

from tokenizer.regex_tokenizer import RegexTokenizer

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------

REPO_ROOT             = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS_FILE   = REPO_ROOT / "data" / "raw" / "corpus_mixture.txt"
DEFAULT_RAW_MANIFEST  = REPO_ROOT / "data" / "raw" / "manifest.json"
DEFAULT_EVAL_MANIFEST = REPO_ROOT / "data" / "eval" / "manifest.json"
SHARD_SIZE_DEFAULT    = 100_000_000   # 100 M tokens → ~100 shards for 10 B tokens

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tokenize corpus_mixture.txt into pretraining shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corpus-file",   type=Path,  default=DEFAULT_CORPUS_FILE)
    p.add_argument("--raw-manifest",  type=Path,  default=DEFAULT_RAW_MANIFEST)
    p.add_argument("--eval-manifest", type=Path,  default=DEFAULT_EVAL_MANIFEST)
    p.add_argument("--tokenizer",     type=str,   default="gpt2",
                   choices=["gpt2", "mgpt2"])
    p.add_argument("--shard-size",    type=int,   default=SHARD_SIZE_DEFAULT,
                   help="Tokens per shard (last shard may be smaller).")
    p.add_argument("--val-split",     type=float, default=0.02,
                   help="Fraction of non-eval lines held out for validation.")
    p.add_argument("--shards-dir",    type=Path,  default=None,
                   help="Output directory. Defaults to data/shards_{tokenizer}/.")
    p.add_argument("--workers",       type=int,   default=max(1, mp.cpu_count() // 2),
                   help="Parallel tokenization workers.")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Multiprocessing worker (must be module-level to be picklable)
# ---------------------------------------------------------------------------

_WORKER_ENCODE_FN = None   # set per-worker via Pool initializer

def _worker_init(encode_fn) -> None:
    global _WORKER_ENCODE_FN
    _WORKER_ENCODE_FN = encode_fn

def _worker_tokenize(text: str) -> list[int]:
    return _WORKER_ENCODE_FN(text)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifests(raw_path: Path, eval_path: Path) -> tuple[int, int, int]:
    """Return (total_lines, eval_line_start, eval_line_count)."""
    raw = json.loads(raw_path.read_text())
    evl = json.loads(eval_path.read_text())
    return raw["lines_written"], evl["eval_line_start"], evl["eval_line_count"]


def _load_tokenizer(name: str) -> tuple:
    """Return (encode_fn, eot_id).

    encode_fn(text: str) -> list[int]  — encodes plain text, no special tokens.
    eot_id                             — integer ID of <|endoftext|>.
    """
    if name == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        eot = enc._special_tokens["<|endoftext|>"]   # 50256
        return enc.encode_ordinary, eot
    else:
        tok = RegexTokenizer()
        tok.load(str(REPO_ROOT / "tokenizer" / "artifacts" / "mgpt2.model"))
        eot = tok.special_tokens["<|endoftext|>"]
        return (lambda text: tok.encode(text, allowed_special=set())), eot


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_shard(buf: np.ndarray, idx: int, split: str, shards_dir: Path) -> None:
    path = shards_dir / f"{split}_{idx:06d}.npy"
    np.save(path, buf)
    LOGGER.info("  wrote %-30s  (%d tokens)", path.name, len(buf))

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _make_shards(
    corpus_file: Path,
    total_lc: int,
    eval_start: int,
    eval_size: int,
    encode_fn,
    eot: int,
    shard_size: int,
    val_split: float,
    shards_dir: Path,
    workers: int = 1,
) -> tuple[int, int, int]:
    """Tokenize and shard the corpus.

    Returns:
        (n_train_shards, n_val_shards, total_tokens)
    """
    shards_dir.mkdir(parents=True, exist_ok=True)

    available       = total_lc - eval_size
    val_doc_count   = round(val_split * available)
    train_doc_count = available - val_doc_count

    LOGGER.info(
        "Corpus: %d total lines | %d eval (skipped) | %d train | %d val | %d workers",
        total_lc, eval_size, train_doc_count, val_doc_count, workers,
    )

    # Single pre-allocated buffer — reused across all shards.
    buf           = np.empty((shard_size,), dtype=np.int32)
    buf_count     = 0
    n_train       = 0
    n_val         = 0
    total_tokens  = 0
    current_split = "train"
    non_eval_seen = 0

    def _line_stream():
        """Yield (absolute_line_idx, stripped_text) for every non-eval line."""
        with corpus_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if eval_start <= i < eval_start + eval_size:
                    continue
                yield i, line.rstrip("\n")

    def _pack(doc_tokens: np.ndarray) -> None:
        """Pack doc_tokens into buf, flushing full shards as needed."""
        nonlocal buf_count, n_train, n_val, total_tokens, current_split
        pos = 0
        while pos < len(doc_tokens):
            space = shard_size - buf_count
            take  = min(space, len(doc_tokens) - pos)
            buf[buf_count:buf_count + take] = doc_tokens[pos:pos + take]
            buf_count    += take
            total_tokens += take
            pos          += take
            if buf_count == shard_size:
                if current_split == "train":
                    _write_shard(buf, n_train, "train", shards_dir)
                    n_train += 1
                else:
                    _write_shard(buf, n_val, "val", shards_dir)
                    n_val += 1
                buf_count = 0

    stream = _line_stream()

    with mp.Pool(workers, initializer=_worker_init, initargs=(encode_fn,)) as pool:
        # imap preserves order and streams lazily — no full corpus in RAM.
        # chunksize=64 amortises IPC overhead without large latency spikes.
        texts_iter  = (text for _, text in stream)
        tokens_iter = pool.imap(_worker_tokenize, texts_iter, chunksize=64)

        # We need the absolute line index alongside each token list to know
        # when to flip train→val.  Re-derive non_eval_seen by counting.
        for ids in tqdm(tokens_iter, total=available, desc="Tokenizing", unit="docs"):
            non_eval_seen += 1

            # Detect train→val boundary; flush train buffer before switching.
            new_split = "val" if non_eval_seen > train_doc_count else "train"
            if new_split != current_split and buf_count > 0:
                _write_shard(buf[:buf_count], n_train, current_split, shards_dir)
                n_train  += 1
                buf_count = 0
            current_split = new_split

            doc_tokens = np.array([eot] + ids, dtype=np.int32)
            _pack(doc_tokens)

    # Flush remainder.
    if buf_count > 0:
        if current_split == "train":
            _write_shard(buf[:buf_count], n_train, "train", shards_dir)
            n_train += 1
        else:
            _write_shard(buf[:buf_count], n_val, "val", shards_dir)
            n_val += 1

    return n_train, n_val, total_tokens


def _write_manifest(
    shards_dir: Path,
    tokenizer_name: str,
    corpus_file: Path,
    eval_start: int,
    eval_size: int,
    val_doc_count: int,
    val_split: float,
    shard_size: int,
    n_train_shards: int,
    n_val_shards: int,
    total_tokens: int,
) -> None:
    if tokenizer_name == "gpt2":
        artifact = {"encoding": "tiktoken/gpt2"}
    else:
        model_path = REPO_ROOT / "tokenizer" / "artifacts" / "mgpt2.model"
        artifact   = {
            "path":   str(model_path.relative_to(REPO_ROOT)),
            "sha256": _sha256(model_path),
        }

    manifest = {
        "tokenizer":          tokenizer_name,
        "tokenizer_artifact": artifact,
        "source_file":        str(corpus_file.relative_to(REPO_ROOT)),
        "eval_line_start":    eval_start,
        "eval_line_count":    eval_size,
        "val_doc_count":      val_doc_count,
        "val_split":          val_split,
        "shard_size_tokens":  shard_size,
        "dtype":              "int32",
        "n_train_shards":     n_train_shards,
        "n_val_shards":       n_val_shards,
        "total_tokens":       total_tokens,
    }
    path = shards_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote manifest → %s", path)

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    shards_dir = args.shards_dir or REPO_ROOT / "data" / f"shards_{args.tokenizer}"

    total_lc, eval_start, eval_size = _load_manifests(args.raw_manifest, args.eval_manifest)
    encode_fn, eot = _load_tokenizer(args.tokenizer)

    available     = total_lc - eval_size
    val_doc_count = round(args.val_split * available)

    n_train, n_val, total_tokens = _make_shards(
        corpus_file=args.corpus_file,
        total_lc=total_lc,
        eval_start=eval_start,
        eval_size=eval_size,
        encode_fn=encode_fn,
        eot=eot,
        shard_size=args.shard_size,
        val_split=args.val_split,
        shards_dir=shards_dir,
        workers=args.workers,
    )

    _write_manifest(
        shards_dir=shards_dir,
        tokenizer_name=args.tokenizer,
        corpus_file=args.corpus_file,
        eval_start=eval_start,
        eval_size=eval_size,
        val_doc_count=val_doc_count,
        val_split=args.val_split,
        shard_size=args.shard_size,
        n_train_shards=n_train,
        n_val_shards=n_val,
        total_tokens=total_tokens,
    )

    LOGGER.info("Done — %d train shards, %d val shards, %d total tokens",
                n_train, n_val, total_tokens)


if __name__ == "__main__":
    main()
