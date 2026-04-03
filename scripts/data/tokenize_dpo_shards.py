"""Tokenize DPO JSONL data into fixed-length shard arrays.

Reads data/dpo/train.jsonl and data/dpo/val.jsonl (after generate_dpo_rejected.py
has been run) and produces three parallel arrays per shard:

    {split}_{index:06d}_chosen.npy       — (N, SEQ_LEN) int32   prompt + chosen response + EOT + padding
    {split}_{index:06d}_rejected.npy     — (N, SEQ_LEN) int32   prompt + rejected response + EOT + padding
    {split}_{index:06d}_prompt_lens.npy  — (N,)         int32   number of prompt tokens per pair

Sequence layout per example:

    [ prompt_tokens … | response_tokens … | EOT | EOT … EOT ]
      ←—— prompt_len ——→ ←—————— response + EOT ——————→ ←pad→

Note: the prompt is NOT followed by EOT — it flows directly into the response.
EOT is appended only at the end of the response.  Padding positions also use EOT
(token ID 50256) but are excluded from loss by the DPO trainer using
prompt_lens[i] + the first EOT position after prompt_lens[i].

prompt_lens[i] tells the trainer exactly where the response starts in both
chosen[i] and rejected[i].  It is identical for both sides of a pair since
they share the same prompt prefix.

Pair alignment:
    chosen[i], rejected[i], and prompt_lens[i] always refer to the same prompt.
    If an example must be skipped (prompt too long, or either response leaves no
    room), BOTH sides are skipped so indices remain aligned.

Pre-condition:
    data/dpo/manifest.json must have "rejected_populated": true.
    The script asserts this and exits early if the flag is absent.
    Run scripts/generate_dpo_rejected.py first.

Usage:
    python scripts/data/tokenize_dpo_shards.py
    python scripts/data/tokenize_dpo_shards.py --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tokenizer.regex_tokenizer import RegexTokenizer

LOGGER = logging.getLogger(__name__)

DEFAULT_DPO_DIR    = REPO_ROOT / "data" / "dpo"
DEFAULT_SHARDS_DIR = REPO_ROOT / "data" / "shards_dpo"
SEQ_LEN_DEFAULT    = 1024
SHARD_EXAMPLES     = 1_000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tokenize DPO JSONL into fixed-length shard arrays (mgpt2 only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dpo-dir",        type=Path, default=DEFAULT_DPO_DIR)
    p.add_argument("--shards-dir",     type=Path, default=DEFAULT_SHARDS_DIR)
    p.add_argument("--seq-len",        type=int,  default=SEQ_LEN_DEFAULT)
    p.add_argument("--shard-examples", type=int,  default=SHARD_EXAMPLES)
    p.add_argument("--workers",        type=int,
                   default=max(1, mp.cpu_count() // 2))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Multiprocessing workers
# ---------------------------------------------------------------------------

_WORKER_ENCODE_FN = None
_WORKER_EOT: int  = 50256
_WORKER_SEQ: int  = SEQ_LEN_DEFAULT


def _worker_init(encode_fn, eot: int, seq_len: int) -> None:
    global _WORKER_ENCODE_FN, _WORKER_EOT, _WORKER_SEQ
    _WORKER_ENCODE_FN = encode_fn
    _WORKER_EOT       = eot
    _WORKER_SEQ       = seq_len


def _pack_sequence(
    prompt_ids: list[int],
    response_ids: list[int],
    eot: int,
    seq_len: int,
) -> list[int] | None:
    """Pack one (prompt, response) pair into a fixed-length token sequence.

    Layout: [prompt | response | EOT | EOT-padding...]
    The prompt is NOT followed by EOT — it flows directly into the response.
    Returns None if the prompt alone fills the budget (no room for any response).
    """
    max_resp = seq_len - len(prompt_ids) - 1   # -1 for the response-terminating EOT
    if max_resp <= 0:
        return None
    if len(response_ids) > max_resp:
        response_ids = response_ids[:max_resp]

    pad = seq_len - len(prompt_ids) - len(response_ids) - 1
    return prompt_ids + response_ids + [eot] + [eot] * pad


def _worker_tokenize_pair(example: dict) -> tuple[list[int], list[int], int] | None:
    """Tokenize one DPO example into (chosen_seq, rejected_seq, prompt_len).

    Returns None if the example must be skipped (prompt too long, or either
    response leaves no room).  Both sides are skipped together so chosen/
    rejected indices stay aligned across all three output arrays.
    """
    encode  = _WORKER_ENCODE_FN
    eot     = _WORKER_EOT
    seq_len = _WORKER_SEQ

    prompt_ids   = encode(example["prompt"])
    chosen_ids   = encode(example["chosen"])
    rejected_ids = encode(example["rejected"])

    chosen_seq   = _pack_sequence(prompt_ids, chosen_ids,   eot, seq_len)
    rejected_seq = _pack_sequence(prompt_ids, rejected_ids, eot, seq_len)

    if chosen_seq is None or rejected_seq is None:
        return None   # skip both sides to keep alignment

    return chosen_seq, rejected_seq, len(prompt_ids)


# ---------------------------------------------------------------------------
# Tokenizer loader
# ---------------------------------------------------------------------------

def _load_tokenizer():
    model_path = REPO_ROOT / "tokenizer" / "artifacts" / "mgpt2.model"
    tok = RegexTokenizer()
    tok.load(str(model_path))
    eot = tok.special_tokens["<|endoftext|>"]
    return (lambda text: tok.encode(text, allowed_special=set())), eot


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_relative(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path.resolve()


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------

def _write_shard(
    chosen_buf:   np.ndarray,
    rejected_buf: np.ndarray,
    lens_buf:     np.ndarray,
    n: int,
    idx: int,
    split: str,
    shards_dir: Path,
) -> None:
    """Write the three arrays for one shard (first n rows/elements only)."""
    np.save(shards_dir / f"{split}_{idx:06d}_chosen.npy",      chosen_buf[:n])
    np.save(shards_dir / f"{split}_{idx:06d}_rejected.npy",    rejected_buf[:n])
    np.save(shards_dir / f"{split}_{idx:06d}_prompt_lens.npy", lens_buf[:n])
    LOGGER.info("  wrote shard %s_%06d (%d pairs)", split, idx, n)


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
    """Tokenize one split into DPO shards.

    Returns (n_shards, n_pairs_written, n_skipped).
    """
    shards_dir.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    LOGGER.info("%s: %d pairs → tokenizing with %d workers", split, len(examples), workers)

    # Pre-allocate three buffers
    chosen_buf   = np.empty((shard_examples, seq_len), dtype=np.int32)
    rejected_buf = np.empty((shard_examples, seq_len), dtype=np.int32)
    lens_buf     = np.empty((shard_examples,),         dtype=np.int32)

    buf_count = n_shards = n_written = n_skipped = 0

    with mp.Pool(workers, initializer=_worker_init,
                 initargs=(encode_fn, eot, seq_len)) as pool:
        results = pool.imap(_worker_tokenize_pair, examples, chunksize=64)

        for result in tqdm(results, total=len(examples), desc=f"  {split}", unit="pair"):
            if result is None:
                n_skipped += 1
                continue

            chosen_seq, rejected_seq, prompt_len = result
            chosen_buf[buf_count]   = chosen_seq
            rejected_buf[buf_count] = rejected_seq
            lens_buf[buf_count]     = prompt_len
            buf_count += 1
            n_written += 1

            if buf_count == shard_examples:
                _write_shard(chosen_buf, rejected_buf, lens_buf,
                             buf_count, n_shards, split, shards_dir)
                n_shards += 1
                buf_count = 0

    if buf_count > 0:
        _write_shard(chosen_buf, rejected_buf, lens_buf,
                     buf_count, n_shards, split, shards_dir)
        n_shards += 1

    if n_skipped:
        LOGGER.warning("%s: skipped %d pairs (prompt too long)", split, n_skipped)

    return n_shards, n_written, n_skipped


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(
    shards_dir: Path,
    dpo_dir: Path,
    dpo_manifest: dict,
    seq_len: int,
    shard_examples: int,
    n_train_shards: int,  n_val_shards: int,
    n_train_pairs: int,   n_val_pairs: int,
    n_skipped: int,
) -> None:
    model_path = REPO_ROOT / "tokenizer" / "artifacts" / "mgpt2.model"
    manifest = {
        "source":              "ai4bharat/IndicAlign (toxic split)",
        "tokenizer":           "mgpt2",
        "tokenizer_artifact":  {
            "path":   str(model_path.relative_to(REPO_ROOT)),
            "sha256": _sha256(model_path),
        },
        "source_train":        str(_try_relative(dpo_dir / "train.jsonl")),
        "source_val":          str(_try_relative(dpo_dir / "val.jsonl")),
        "rejected_source":     dpo_manifest.get("rejected_source", "unknown"),
        "rejected_model_step": dpo_manifest.get("rejected_model_step", -1),
        "max_seq_len":         seq_len,
        "pad_token":           50256,
        "shard_examples":      shard_examples,
        "n_train_shards":      n_train_shards,
        "n_val_shards":        n_val_shards,
        "n_train_pairs":       n_train_pairs,
        "n_val_pairs":         n_val_pairs,
        "n_skipped":           n_skipped,
        "dtype":               "int32",
        "arrays_per_shard":    ["chosen", "rejected", "prompt_lens"],
        "chosen_rejected_shape": f"(N_pairs_in_shard, {seq_len})",
        "prompt_lens_shape":     "(N_pairs_in_shard,)",
    }
    path = shards_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Manifest → %s", path)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    # ── pre-condition check ───────────────────────────────────────────────
    dpo_manifest_path = args.dpo_dir / "manifest.json"
    dpo_manifest      = json.loads(dpo_manifest_path.read_text(encoding="utf-8"))

    if not dpo_manifest.get("rejected_populated", False):
        print(
            "ERROR: data/dpo/manifest.json has rejected_populated=false.\n"
            "Run scripts/generate_dpo_rejected.py first to populate the "
            "'rejected' field in all DPO examples.",
            file=sys.stderr,
        )
        sys.exit(1)

    LOGGER.info("rejected_populated=true — proceeding with tokenization")

    encode_fn, eot = _load_tokenizer()
    total_skipped  = 0

    n_train_shards, n_train_pairs, sk = _shard_split(
        jsonl_path    = args.dpo_dir / "train.jsonl",
        split         = "train",
        encode_fn     = encode_fn,
        eot           = eot,
        seq_len       = args.seq_len,
        shard_examples= args.shard_examples,
        shards_dir    = args.shards_dir,
        workers       = args.workers,
    )
    total_skipped += sk

    n_val_shards, n_val_pairs, sk = _shard_split(
        jsonl_path    = args.dpo_dir / "val.jsonl",
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
        shards_dir     = args.shards_dir,
        dpo_dir        = args.dpo_dir,
        dpo_manifest   = dpo_manifest,
        seq_len        = args.seq_len,
        shard_examples = args.shard_examples,
        n_train_shards = n_train_shards,  n_val_shards = n_val_shards,
        n_train_pairs  = n_train_pairs,   n_val_pairs  = n_val_pairs,
        n_skipped      = total_skipped,
    )

    LOGGER.info(
        "Done — train: %d shards (%d pairs) | val: %d shards (%d pairs) | skipped: %d",
        n_train_shards, n_train_pairs, n_val_shards, n_val_pairs, total_skipped,
    )


if __name__ == "__main__":
    main()
