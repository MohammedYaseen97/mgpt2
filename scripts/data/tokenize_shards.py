import json
import logging
import argparse

from pathlib import Path

import tiktoken
import numpy as np
from tqdm import tqdm

from tokenizer.regex_tokenizer import RegexTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS_FILE = REPO_ROOT / "data" / "raw" / "corpus_mixture.txt"
DEFAULT_RAW_MANIFEST = REPO_ROOT / "data" / "raw" / "manifest.json"
DEFAULT_EVAL_MANIFEST = REPO_ROOT / "data" / "eval" / "manifest.json"
DEFAULT_SHARDS_DIR  = REPO_ROOT / "data" / "shards"

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize shards from corpus_mixture.txt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus-file", type=Path, default=DEFAULT_CORPUS_FILE)
    parser.add_argument("--raw-manifest", type=Path, default=DEFAULT_RAW_MANIFEST)
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--shard-size", type=int, default=1_000_000, help="Number of tokens per shard.")
    parser.add_argument("--tokenizer", type=str, default="gpt2", choices=["gpt2", "mgpt2"])
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _load_manifest(raw_manifest: Path, eval_manifest: Path) -> tuple[int, int, int]:
    with open(raw_manifest, "r", encoding="utf-8") as f:
        raw_manifest = json.load(f)
    with open(eval_manifest, "r", encoding="utf-8") as f:
        eval_manifest = json.load(f)
    return raw_manifest["lines_written"], eval_manifest["eval_line_start"], eval_manifest["eval_line_count"]

def _load_tokenizer(tokenizer_name: str):
    if tokenizer_name == "gpt2":
        # tiktoken_gpt2 — same library and encoding used in tokenizer/scripts/evaluate.py
        # and in Karpathy's fineweb.py, so the controlled baseline is fully traceable.
        return tiktoken.get_encoding("gpt2")
    elif tokenizer_name == "mgpt2":
        # RegexTokenizer.load() from the canonical local artifact — identical load path
        # to tokenizer/scripts/evaluate.py; no network dependency during sharding.
        tokenizer = RegexTokenizer()
        tokenizer.load(str(REPO_ROOT / "tokenizer" / "artifacts" / "mgpt2.model"))
        return tokenizer
    else:
        raise ValueError(f"Invalid tokenizer: {tokenizer_name}")

def _write_shard(tokens: list[int], shard_idx: int):
    shard_file = DEFAULT_SHARDS_DIR / f"shard_{shard_idx:06d}.npy"
    np.save(shard_file, np.array(tokens, dtype=np.int32))

def _make_shards(corpus_file: Path, total_size: int, eval_start: int, eval_size: int, tokenizer: RegexTokenizer, shard_size: int):
    LOGGER.info("Tokenizing shards from %s (total_size=%d, eval_start=%d, eval_size=%d)", corpus_file, total_size, eval_start, eval_size)
    with open(corpus_file, "r", encoding="utf-8") as f:
        current_tokens = []
        shard_idx = 0
        
        val_lc = 0.02
        
        for i, line in enumerate(f):
            if i >= eval_start and i < eval_start + eval_size:
                continue
            
            line = line.rstrip("\n")
            tokens = tokenizer.encode(line) + tokenizer.encode("<|endoftext|>")
            current_tokens.extend(tokens)
            if len(current_tokens) >= shard_size:
                _write_shard(current_tokens, shard_idx)
                current_tokens = []
                shard_idx += 1
                
        if current_tokens:
            _write_shard(current_tokens, shard_idx)
        
        return shard_idx + 1

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    
    total_size, eval_start, eval_size = _load_manifest(args.raw_manifest, args.eval_manifest)
    
    tokenizer = _load_tokenizer(args.tokenizer)
    shard_count = _make_shards(args.corpus_file, total_size, eval_start, eval_size, tokenizer, args.shard_size)
    
    LOGGER.info("Made %d shards", shard_count)

if __name__ == "__main__":
    main()