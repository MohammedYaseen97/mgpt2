"""Tokenize SFT JSONL data into fixed-length shard arrays.

Reads data/sft/train.jsonl and data/sft/val.jsonl (from build_sft_data.py) and
produces parallel tokens.npy + mask.npy shards under data/shards_sft/.

Tokenizer
---------
mgpt2 only — the gpt2-tokenized model is retired after Phase C.  Phase D's
controlled baseline is the same mgpt2 pretrained model without SFT.

Sequence layout per example (seq_len = 1024):

    [prompt_tokens … | response_tokens … | EOT | EOT … EOT]
     mask:  0 … 0     1 … 1               1     0 … 0

    prompt tokens  — mask = 0  (loss ignored)
    response tokens— mask = 1  (loss computed)
    EOT (end of response) — mask = 1
    padding (EOT fill)    — mask = 0

Padding token: EOT = 50256.  Token 0 is a real vocabulary token and must not
be used as a padding sentinel.

Truncation: if prompt + response + 1 (EOT) > seq_len, the response is trimmed
from the right until it fits.  If prompt alone leaves no room for even a single
response token + EOT, the example is skipped with a warning.

Output shape per shard:
    {split}_{idx:06d}_tokens.npy  — np.int32, shape (N, SEQ_LEN)
    {split}_{idx:06d}_mask.npy    — np.int32, shape (N, SEQ_LEN)

Memory design:
    Two pre-allocated buffers of shape (shard_examples, seq_len) in int32.
    RAM cost: 2 × 1000 × 1024 × 4 bytes ≈ 8 MB — trivial.
    Workers return Python lists of length seq_len through IPC; numpy conversion
    happens in the main process only at flush time.
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
from tqdm import tqdm

from tokenizer.regex_tokenizer import RegexTokenizer

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------

REPO_ROOT              = Path(__file__).resolve().parent.parent.parent
DEFAULT_SFT_DIR        = REPO_ROOT / "data" / "sft"
DEFAULT_SHARDS_DIR     = REPO_ROOT / "data" / "shards_sft"
DEFAULT_SFT_MANIFEST   = DEFAULT_SFT_DIR / "manifest.json"

SEQ_LEN_DEFAULT        = 1024
SHARD_EXAMPLES_DEFAULT = 1_000   # examples per shard; each is SEQ_LEN tokens

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tokenize SFT JSONL into fixed-length shard arrays (mgpt2 only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sft-dir",       type=Path, default=DEFAULT_SFT_DIR,
                   help="Directory containing train.jsonl, val.jsonl, manifest.json.")
    p.add_argument("--shards-dir",    type=Path, default=DEFAULT_SHARDS_DIR)
    p.add_argument("--seq-len",       type=int,  default=SEQ_LEN_DEFAULT,
                   help="Fixed sequence length; shorter examples are padded, longer truncated.")
    p.add_argument("--shard-examples",type=int,  default=SHARD_EXAMPLES_DEFAULT,
                   help="Examples per shard (last shard may be smaller).")
    p.add_argument("--workers",       type=int,  default=max(1, mp.cpu_count() // 2),
                   help="Parallel tokenization workers.")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Multiprocessing workers — module-level so they are picklable
# ---------------------------------------------------------------------------

_WORKER_ENCODE_FN = None
_WORKER_EOT:       int = 50256
_WORKER_SEQ_LEN:   int = SEQ_LEN_DEFAULT


def _worker_init(encode_fn, eot: int, seq_len: int) -> None:
    global _WORKER_ENCODE_FN, _WORKER_EOT, _WORKER_SEQ_LEN
    _WORKER_ENCODE_FN = encode_fn
    _WORKER_EOT       = eot
    _WORKER_SEQ_LEN   = seq_len


def _worker_tokenize_example(
    example: dict,
) -> tuple[list[int], list[int]] | None:
    """Tokenize one SFT example into (tokens, mask) both of length seq_len.

    Returns None if the prompt alone exceeds the available budget.
    """
    encode  = _WORKER_ENCODE_FN
    eot     = _WORKER_EOT
    seq_len = _WORKER_SEQ_LEN

    prompt_ids = encode(example["prompt"])
    resp_ids   = encode(example["response"])

    # Maximum response tokens = seq_len - prompt - 1 EOT
    max_resp = seq_len - len(prompt_ids) - 1
    if max_resp <= 0:
        return None   # prompt too long; skip

    if len(resp_ids) > max_resp:
        resp_ids = resp_ids[:max_resp]

    prompt_len = len(prompt_ids)
    resp_len   = len(resp_ids)
    pad_count  = seq_len - prompt_len - resp_len - 1   # -1 for the EOT

    tokens = prompt_ids + resp_ids + [eot] + [eot] * pad_count
    mask   = [0] * prompt_len + [1] * resp_len + [1] + [0] * pad_count

    return tokens, mask

# ---------------------------------------------------------------------------
# Tokenizer loader (mgpt2 only — identical to tokenize_shards.py)
# ---------------------------------------------------------------------------

def _load_tokenizer() -> tuple:
    """Return (encode_fn, eot_id) for mgpt2."""
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

# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------

def _write_shard_pair(
    tokens_buf: np.ndarray,
    mask_buf:   np.ndarray,
    n: int,
    idx: int,
    split: str,
    shards_dir: Path,
) -> None:
    """Write tokens_buf[:n] and mask_buf[:n] as paired .npy files."""
    t_path = shards_dir / f"{split}_{idx:06d}_tokens.npy"
    m_path = shards_dir / f"{split}_{idx:06d}_mask.npy"
    np.save(t_path, tokens_buf[:n])
    np.save(m_path, mask_buf[:n])
    LOGGER.info("  wrote %s + _mask  (%d examples)", t_path.name, n)

# ---------------------------------------------------------------------------
# Core sharding loop
# ---------------------------------------------------------------------------

def _shard_split(
    jsonl_path: Path,
    split: str,
    encode_fn,
    eot: int,
    seq_len: int,
    shard_examples: int,
    shards_dir: Path,
    workers: int,
) -> tuple[int, int, int]:
    """Tokenize one JSONL split into shards.

    Returns:
        (n_shards, n_examples_written, n_skipped)
    """
    shards_dir.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    total      = len(examples)
    LOGGER.info("%s: %d examples → tokenizing with %d workers", split, total, workers)

    # Pre-allocated buffers — reused across shards.
    tokens_buf = np.empty((shard_examples, seq_len), dtype=np.int32)
    mask_buf   = np.empty((shard_examples, seq_len), dtype=np.int32)
    buf_count  = 0
    n_shards   = 0
    n_written  = 0
    n_skipped  = 0

    with mp.Pool(workers, initializer=_worker_init,
                 initargs=(encode_fn, eot, seq_len)) as pool:
        results = pool.imap(_worker_tokenize_example, examples, chunksize=64)

        for result in tqdm(results, total=total, desc=f"  {split}", unit="ex"):
            if result is None:
                n_skipped += 1
                continue

            tokens, mask = result
            tokens_buf[buf_count] = tokens
            mask_buf[buf_count]   = mask
            buf_count += 1
            n_written += 1

            if buf_count == shard_examples:
                _write_shard_pair(tokens_buf, mask_buf, buf_count,
                                  n_shards, split, shards_dir)
                n_shards  += 1
                buf_count  = 0

    # Flush remainder.
    if buf_count > 0:
        _write_shard_pair(tokens_buf, mask_buf, buf_count,
                          n_shards, split, shards_dir)
        n_shards += 1

    if n_skipped:
        LOGGER.warning("%s: skipped %d examples (prompt too long)", split, n_skipped)

    return n_shards, n_written, n_skipped

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(
    shards_dir: Path,
    sft_dir: Path,
    seq_len: int,
    shard_examples: int,
    n_train_shards: int,
    n_val_shards: int,
    n_train_examples: int,
    n_val_examples: int,
    n_skipped: int,
) -> None:
    model_path = REPO_ROOT / "tokenizer" / "artifacts" / "mgpt2.model"
    manifest = {
        "source":             "ai4bharat/indic-align (Dolly_T + OpenAssistant_T + Anudesh)",
        "tokenizer":          "mgpt2",
        "tokenizer_artifact": {
            "path":   str(model_path.relative_to(REPO_ROOT)),
            "sha256": _sha256(model_path),
        },
        "seed":               42,
        "n_train_examples":   n_train_examples,
        "n_val_examples":     n_val_examples,
        "max_seq_len":        seq_len,
        "n_train_shards":     n_train_shards,
        "n_val_shards":       n_val_shards,
        "n_skipped":          n_skipped,
        "dtype":              "int32",
    }
    path = shards_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Manifest → %s", path)

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    encode_fn, eot = _load_tokenizer()

    total_skipped = 0
    n_train_shards, n_train_ex, sk = _shard_split(
        jsonl_path    = args.sft_dir / "train.jsonl",
        split         = "train",
        encode_fn     = encode_fn,
        eot           = eot,
        seq_len       = args.seq_len,
        shard_examples= args.shard_examples,
        shards_dir    = args.shards_dir,
        workers       = args.workers,
    )
    total_skipped += sk

    n_val_shards, n_val_ex, sk = _shard_split(
        jsonl_path    = args.sft_dir / "val.jsonl",
        split         = "val",
        encode_fn     = encode_fn,
        eot           = eot,
        seq_len       = args.seq_len,
        shard_examples= args.shard_examples,
        shards_dir    = args.shards_dir,
        workers       = args.workers,
    )
    total_skipped += sk

    _write_manifest(
        shards_dir      = args.shards_dir,
        sft_dir         = args.sft_dir,
        seq_len         = args.seq_len,
        shard_examples  = args.shard_examples,
        n_train_shards  = n_train_shards,
        n_val_shards    = n_val_shards,
        n_train_examples= n_train_ex,
        n_val_examples  = n_val_ex,
        n_skipped       = total_skipped,
    )

    LOGGER.info(
        "Done — train: %d shards (%d ex) | val: %d shards (%d ex) | skipped: %d",
        n_train_shards, n_train_ex, n_val_shards, n_val_ex, total_skipped,
    )


if __name__ == "__main__":
    main()
