import logging
import argparse

from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS_FILE = REPO_ROOT / "data" / "raw" / "corpus_mixture.txt"
DEFAULT_EVAL_DIR    = REPO_ROOT / "data" / "eval"
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
    parser.add_argument(
        "--validation-size",
        type=int,
        default=50_000,
        help=(
            "Number of lines to use for the validation set. "
            "The validation set is taken from the lines remaining in the corpus after the eval slice is removed, "
            "selecting from the end of this remainder."
        ),
    )
    parser.add_argument("--tokenizer", type=str, default="gpt2", choices=["gpt2", "mgpt2"])
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _load_tokenizer(tokenizer: str):
    if tokenizer == "gpt2":
        return GPT2Tokenizer.from_pretrained("gpt2")
    elif tokenizer == "mgpt2":
        return GPT2Tokenizer.from_pretrained("mgpt2")
    else:
        raise ValueError(f"Invalid tokenizer: {tokenizer}")

def _tokenize_shards(corpus_file: Path, validation_size: int, tokenizer: str):
    LOGGER.info("Tokenizing shards from %s (validation_size=%d)", corpus_file, validation_size)
    with open(corpus_file, "r", encoding="utf-8") as f:
        current_tokens = []
        line_count = 0
        shard_idx = 0
        
        for line in f:
            tokens = tokenizer.encode(line)
            

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    
    tokenizer = _load_tokenizer(args.tokenizer)
    
    _tokenize_shards(args.corpus_file, args.validation_size, args.tokenizer)
    
    _write_manifest(args.corpus_file, args.validation_size, args.tokenizer)

if __name__ == "__main__":
    main()