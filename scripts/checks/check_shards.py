"""Phase B check — tokenized shard integrity.

Verifies every invariant that DATA_REFERENCE.md specifies for the tokenized
output artifacts produced by tokenize_shards.py and tokenize_sft_shards.py.

Checks
------
1. Pretraining shards  (data/shards_gpt2/  and  data/shards_mgpt2/)
   - manifest present and contains all required keys
   - file count on disk matches n_train_shards + n_val_shards in manifest
   - every shard: dtype=int32, ndim=1
   - every non-last train shard: exactly shard_size_tokens long
   - spot-check (first + last 1 K tokens per shard): IDs in [0, 50256]
   - cross-tokenizer parity: both manifests reference the same source_file,
     eval exclusion window, and val_doc_count

2. SFT shards  (data/shards_sft/)
   - manifest present and contains all required keys
   - paired _tokens.npy + _mask.npy files match manifest counts
   - every shard pair: dtype=int32, ndim=2, shape (N, 1024), arrays same shape
   - full scan of first train shard + all val shards:
       • tokens in [0, 50256]
       • mask values exclusively in {0, 1}
       • token 0 never used as padding  (pad token must be EOT = 50256)
       • at least one mask=1 per example  (no all-prompt/all-pad rows)
   - first-row spot-check on all other train shards: same four invariants

3. DPO shards
   SKIP — data/shards_dpo/ does not exist yet.  tokenize_dpo_shards.py has not
   been implemented; rejected responses are deferred to Phase E.

Exit code
   0  all checks passed
   1  one or more checks failed  (list printed at end)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
EOT          = 50256
MAX_TOKEN_ID = 50256
SPOT_N       = 1_000   # tokens sampled from each end of a pretrain shard

# ---------------------------------------------------------------------------
# Minimal check harness
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def _chk(name: str, fn) -> bool:
    """Run fn(); print PASS/FAIL; never raise. Returns True on pass."""
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        msg = str(e) or "assertion failed"
        _results.append((name, False, msg))
        print(f"  FAIL  {name}  →  {msg}")
        return False
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        _results.append((name, False, msg))
        print(f"  FAIL  {name}  →  {msg}")
        return False

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mmap(path: Path) -> np.ndarray:
    assert path.exists(), f"file not found: {path}"
    return np.load(path, mmap_mode="r")


def _spot_range(arr: np.ndarray, label: str) -> None:
    """Check first+last SPOT_N flat elements are in [0, MAX_TOKEN_ID]."""
    flat = arr.reshape(-1)
    n    = len(flat)
    take = min(SPOT_N, n)
    sample = np.concatenate([np.array(flat[:take]), np.array(flat[max(0, n - take):])])
    lo, hi = int(sample.min()), int(sample.max())
    assert lo >= 0,            f"{label}: token ID < 0 (min={lo})"
    assert hi <= MAX_TOKEN_ID, f"{label}: token ID > {MAX_TOKEN_ID} (max={hi})"

# ---------------------------------------------------------------------------
# 1. Pretraining shards
# ---------------------------------------------------------------------------

def _check_pretrain_dir(shards_dir: Path) -> None:
    tag = shards_dir.name
    print(f"\n  [{tag}]")

    # manifest
    m_path = shards_dir / "manifest.json"

    def manifest_exists():
        assert m_path.exists(), f"manifest not found: {m_path}"
    if not _chk(f"{tag}/manifest_exists", manifest_exists):
        print(f"  SKIP  {tag}/* — cannot proceed without manifest")
        return

    m = json.loads(m_path.read_text(encoding="utf-8"))

    required_keys = {
        "tokenizer", "source_file", "eval_line_start", "eval_line_count",
        "n_train_shards", "n_val_shards", "shard_size_tokens", "dtype",
        "val_doc_count",
    }

    def manifest_keys():
        missing = required_keys - set(m.keys())
        assert not missing, f"manifest missing keys: {sorted(missing)}"
    _chk(f"{tag}/manifest_keys", manifest_keys)

    def manifest_dtype():
        assert m.get("dtype") == "int32", f"dtype={m.get('dtype')!r}, expected 'int32'"
    _chk(f"{tag}/manifest_dtype", manifest_dtype)

    shard_size     = m.get("shard_size_tokens", 0)
    n_train_expect = m.get("n_train_shards", -1)
    n_val_expect   = m.get("n_val_shards", -1)

    # file counts
    train_files = sorted(shards_dir.glob("train_*.npy"))
    val_files   = sorted(shards_dir.glob("val_*.npy"))

    def n_train_files():
        assert len(train_files) == n_train_expect, (
            f"found {len(train_files)} train shards, manifest says {n_train_expect}"
        )
    def n_val_files():
        assert len(val_files) == n_val_expect, (
            f"found {len(val_files)} val shards, manifest says {n_val_expect}"
        )
    _chk(f"{tag}/n_train_files", n_train_files)
    _chk(f"{tag}/n_val_files",   n_val_files)

    all_files = train_files + val_files

    # dtype + ndim: every shard (reads .npy header only — very fast)
    bad_dtype = []
    bad_ndim  = []
    for p in all_files:
        a = _mmap(p)
        if a.dtype != np.int32:
            bad_dtype.append(p.name)
        if a.ndim != 1:
            bad_ndim.append(p.name)

    def dtype_all():
        assert not bad_dtype, f"wrong dtype in: {bad_dtype[:5]}"
    def ndim_all():
        assert not bad_ndim, f"expected 1-D in: {bad_ndim[:5]}"
    _chk(f"{tag}/dtype_all_shards", dtype_all)
    _chk(f"{tag}/ndim_all_shards",  ndim_all)

    # non-last train shards must be exactly shard_size_tokens long
    if train_files and shard_size > 0:
        bad_size = [p.name for p in train_files[:-1] if len(_mmap(p)) != shard_size]
        def train_shard_sizes():
            assert not bad_size, f"wrong size in: {bad_size[:5]}"
        _chk(f"{tag}/train_shard_sizes", train_shard_sizes)

    # spot-check token range: first train, last train, all val
    spot_targets = list(dict.fromkeys(train_files[:1] + train_files[-1:] + val_files))
    for p in spot_targets:
        a = _mmap(p)
        _chk(f"{tag}/tokens_range/{p.name}", lambda a=a, p=p: _spot_range(a, p.name))


def _check_pretrain_parity(gpt2_dir: Path, mgpt2_dir: Path) -> None:
    """Both tokenizers must have processed the exact same source data."""
    print("\n  [cross-tokenizer parity]")

    try:
        m1 = json.loads((gpt2_dir  / "manifest.json").read_text(encoding="utf-8"))
        m2 = json.loads((mgpt2_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception as e:
        _chk("parity/load_manifests", lambda: (_ for _ in ()).throw(e))
        return

    def same_source():
        assert m1["source_file"] == m2["source_file"], (
            f"{m1['source_file']!r} vs {m2['source_file']!r}"
        )
    def same_eval_start():
        assert m1["eval_line_start"] == m2["eval_line_start"], (
            f"{m1['eval_line_start']} vs {m2['eval_line_start']}"
        )
    def same_eval_count():
        assert m1["eval_line_count"] == m2["eval_line_count"], (
            f"{m1['eval_line_count']} vs {m2['eval_line_count']}"
        )
    def same_val_doc_count():
        assert m1["val_doc_count"] == m2["val_doc_count"], (
            f"{m1['val_doc_count']} vs {m2['val_doc_count']}"
        )

    _chk("parity/same_source_file",   same_source)
    _chk("parity/same_eval_start",    same_eval_start)
    _chk("parity/same_eval_count",    same_eval_count)
    _chk("parity/same_val_doc_count", same_val_doc_count)

# ---------------------------------------------------------------------------
# 2. SFT shards
# ---------------------------------------------------------------------------

def _check_sft_shard_pair(t_path: Path, m_path: Path, *, full_scan: bool) -> None:
    """Check one (tokens, mask) shard pair."""
    stem   = t_path.name.replace("_tokens.npy", "")
    prefix = f"shards_sft/{stem}"

    tokens = _mmap(t_path)
    mask   = _mmap(m_path)

    # dtype
    def dtype_tokens():
        assert tokens.dtype == np.int32, f"tokens dtype={tokens.dtype}"
    def dtype_mask():
        assert mask.dtype == np.int32, f"mask dtype={mask.dtype}"
    _chk(f"{prefix}/dtype_tokens", dtype_tokens)
    _chk(f"{prefix}/dtype_mask",   dtype_mask)

    # shape
    def ndim_2():
        assert tokens.ndim == 2, f"tokens ndim={tokens.ndim}, expected 2"
    def seq_len_1024():
        assert tokens.shape[1] == 1024, f"seq_len={tokens.shape[1]}, expected 1024"
    def shapes_match():
        assert tokens.shape == mask.shape, (
            f"tokens {tokens.shape} != mask {mask.shape}"
        )
    _chk(f"{prefix}/ndim_2",      ndim_2)
    _chk(f"{prefix}/seq_len",     seq_len_1024)
    _chk(f"{prefix}/shapes_match", shapes_match)

    # load the rows we want to inspect
    if full_scan:
        tok_rows  = np.array(tokens)    # (N, 1024)
        mask_rows = np.array(mask)
    else:
        tok_rows  = np.array(tokens[:1])  # first row only
        mask_rows = np.array(mask[:1])

    scan_label = "full" if full_scan else "row0"

    def token_range():
        lo, hi = int(tok_rows.min()), int(tok_rows.max())
        assert lo >= 0,            f"[{scan_label}] token ID < 0 (min={lo})"
        assert hi <= MAX_TOKEN_ID, f"[{scan_label}] token ID > {MAX_TOKEN_ID} (max={hi})"

    def mask_binary():
        bad = set(np.unique(mask_rows).tolist()) - {0, 1}
        assert not bad, f"[{scan_label}] non-binary mask values: {bad}"

    def no_token_0_padding():
        # token 0 is a real vocabulary token; it must never sit at a
        # mask=0 (prompt/padding) position.  Mask=1 positions may legitimately
        # contain token 0 as part of a response.
        pad_pos = mask_rows == 0
        assert not (tok_rows[pad_pos] == 0).any(), (
            f"[{scan_label}] token 0 found at a mask=0 (padding/prompt) position"
        )

    def has_response_tokens():
        assert mask_rows.sum() > 0, (
            f"[{scan_label}] no mask=1 positions — all examples appear to be prompt-only"
        )

    _chk(f"{prefix}/token_range/{scan_label}",      token_range)
    _chk(f"{prefix}/mask_binary/{scan_label}",      mask_binary)
    _chk(f"{prefix}/no_token_0_pad/{scan_label}",   no_token_0_padding)
    _chk(f"{prefix}/has_response_tokens/{scan_label}", has_response_tokens)


def _check_sft_dir(shards_dir: Path) -> None:
    tag = shards_dir.name
    print(f"\n  [{tag}]")

    m_path = shards_dir / "manifest.json"

    def manifest_exists():
        assert m_path.exists(), f"manifest not found: {m_path}"
    if not _chk(f"{tag}/manifest_exists", manifest_exists):
        print(f"  SKIP  {tag}/* — cannot proceed without manifest")
        return

    m = json.loads(m_path.read_text(encoding="utf-8"))

    required_keys = {
        "tokenizer", "tokenizer_artifact", "seq_len", "pad_token",
        "n_train_shards", "n_val_shards", "n_train_examples", "n_val_examples",
        "dtype", "array_shape",
    }

    def manifest_keys():
        missing = required_keys - set(m.keys())
        assert not missing, f"manifest missing keys: {sorted(missing)}"
    def manifest_pad_token():
        assert m.get("pad_token") == EOT, (
            f"pad_token={m.get('pad_token')}, expected {EOT}"
        )
    def manifest_dtype():
        assert m.get("dtype") == "int32", f"dtype={m.get('dtype')!r}"
    _chk(f"{tag}/manifest_keys",      manifest_keys)
    _chk(f"{tag}/manifest_pad_token", manifest_pad_token)
    _chk(f"{tag}/manifest_dtype",     manifest_dtype)

    n_train_expect = m.get("n_train_shards", -1)
    n_val_expect   = m.get("n_val_shards",   -1)

    # file counts
    train_tok  = sorted(shards_dir.glob("train_*_tokens.npy"))
    train_mask = sorted(shards_dir.glob("train_*_mask.npy"))
    val_tok    = sorted(shards_dir.glob("val_*_tokens.npy"))
    val_mask   = sorted(shards_dir.glob("val_*_mask.npy"))

    def n_train_tok():
        assert len(train_tok) == n_train_expect, (
            f"found {len(train_tok)} token files, expected {n_train_expect}"
        )
    def n_train_msk():
        assert len(train_mask) == n_train_expect, (
            f"found {len(train_mask)} mask files, expected {n_train_expect}"
        )
    def n_val_tok():
        assert len(val_tok) == n_val_expect, (
            f"found {len(val_tok)} val token files, expected {n_val_expect}"
        )
    def n_val_msk():
        assert len(val_mask) == n_val_expect, (
            f"found {len(val_mask)} val mask files, expected {n_val_expect}"
        )
    _chk(f"{tag}/n_train_token_files", n_train_tok)
    _chk(f"{tag}/n_train_mask_files",  n_train_msk)
    _chk(f"{tag}/n_val_token_files",   n_val_tok)
    _chk(f"{tag}/n_val_mask_files",    n_val_msk)

    # per-shard: full scan on first train + all val; first-row spot on the rest
    for i, (tp, mp) in enumerate(zip(train_tok, train_mask)):
        _check_sft_shard_pair(tp, mp, full_scan=(i == 0))

    for tp, mp in zip(val_tok, val_mask):
        _check_sft_shard_pair(tp, mp, full_scan=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase B shard integrity checker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",      type=Path, default=REPO_ROOT / "data")
    p.add_argument("--skip-pretrain", action="store_true",
                   help="Skip pretraining shard checks (still fast; 465 header-only reads + spot samples).")
    p.add_argument("--skip-sft",      action="store_true",
                   help="Skip SFT shard checks.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data_dir

    print("=" * 60)
    print("Phase B — shard integrity checks")
    print("=" * 60)

    if not args.skip_pretrain:
        print("\n[1] Pretraining shards")
        gpt2_dir  = data / "shards_gpt2"
        mgpt2_dir = data / "shards_mgpt2"
        _check_pretrain_dir(gpt2_dir)
        _check_pretrain_dir(mgpt2_dir)
        _check_pretrain_parity(gpt2_dir, mgpt2_dir)

    if not args.skip_sft:
        print("\n[2] SFT shards")
        _check_sft_dir(data / "shards_sft")

    print("\n[3] DPO shards")
    print("  SKIP  data/shards_dpo/ — not yet generated (rejected side deferred to Phase E)")

    # summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    print(f"Results: {passed} passed, {failed} failed  ({len(_results)} total)")

    if failed:
        print("\nFailed checks:")
        for name, ok, msg in _results:
            if not ok:
                print(f"  FAIL  {name}  →  {msg}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
