"""Build a curated SFT corpus from ai4bharat/indic-align.

Sources
-------
- Anudesh        — crowd-sourced native English Q&A (36.8k rows, English only)
- Dolly_T        — translated Dolly-15K, all 14 Indic languages as columns (15k rows)
- OpenAssistant_T— translated OpenAssistant v1, same schema (19.9k rows)

Schema (Dolly_T / OpenAssistant_T)
-----------------------------------
Each row has one column per language variant, e.g. ``hin_Deva``, ``kan_Knda``,
``hin_Latn``, ``kan_Latn``.  Each column is a string repr of::

    [[prompt, response], [prompt, response], ...]   # multi-turn conversation

Only the first turn is used; subsequent turns are context-dependent and
unsuitable for single-turn SFT at GPT-2 scale.

Schema (Anudesh)
----------------
Each row has an ``interactions`` field with the same [[p,r],...] format but
in English only.

Language distribution
---------------------
Mirrors the pretraining weights so SFT does not shift the language balance:

  eng_Latn  55%  = 16,500   Anudesh
  hin_Deva  18%  =  5,400   Pool partition A
  kan_Knda  13%  =  3,900   Pool partition B
  hin_Latn   7%  =  2,100   Pool partition C
  kan_Latn   7%  =  2,100   Pool partition D
  ─────────────────────────
  Total    100%  = 30,000

Disjoint row partitioning
--------------------------
Dolly_T + OpenAssistant_T rows are pooled, shuffled with a fixed seed, then
sliced into four non-overlapping partitions.  A source row contributes exactly
ONE language column — the same English content never appears in multiple scripts.

Swap correction
---------------
~10% of Dolly_T Latin-script rows have prompt/response swapped (upstream dataset
bug).  The script corrects these by aligning against the eng_Latn column of the
same row: if English has a short prompt + long response but the target column has
a long[0] + short[1], the target's pair is swapped back.

Outputs
-------
  data/sft/train.jsonl   — 90% of examples
  data/sft/val.jsonl     — 10% of examples
  data/sft/manifest.json

Each JSONL line: {"prompt": "...", "response": "...", "lang": "hin_Deva"}
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "sft"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_ID     = "ai4bharat/indic-align"
POOL_CONFIGS   = ["Dolly_T", "OpenAssistant_T"]
ENGLISH_CONFIG = "Anudesh"

# (column_name, target_count)  — order defines partition assignment
LANG_SLOTS = [
    ("hin_Deva", 5_400),
    ("kan_Knda", 3_900),
    ("hin_Latn", 2_100),
    ("kan_Latn", 2_100),
]
ENGLISH_COUNT = 16_500
TOTAL_TARGET  = 30_000
VAL_RATIO     = 0.10

DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a curated SFT JSONL corpus from ai4bharat/indic-align.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed",    type=int,  default=DEFAULT_SEED)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Directory for train.jsonl, val.jsonl, manifest.json")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Turn parsing
# ---------------------------------------------------------------------------

def _parse_turns(raw: Any) -> list[tuple[str, str]]:
    """Parse a language-column value into (prompt, response) pairs.

    Handles both raw Python list/list-of-lists and its string repr.
    Returns an empty list if the value is missing or malformed.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    else:
        parsed = raw

    if not isinstance(parsed, list):
        return []

    result = []
    for turn in parsed:
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            p = str(turn[0]).strip()
            r = str(turn[1]).strip()
            if p and r:
                result.append((p, r))
    return result


def _correct_swap(
    eng_turns: list[tuple[str, str]],
    tgt_turns: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Fix upstream swap bug in Dolly_T Latin columns (~10% of rows).

    If English has a short prompt + long response but the target column
    has long[0] + short[1], the pair is inverted — swap it back.
    """
    if not eng_turns or not tgt_turns:
        return tgt_turns
    ep_len = len(eng_turns[0][0])
    er_len = len(eng_turns[0][1])
    t0_len = len(tgt_turns[0][0])
    t1_len = len(tgt_turns[0][1])
    if er_len > ep_len and t0_len > t1_len:
        fixed = [(tgt_turns[0][1], tgt_turns[0][0])] + list(tgt_turns[1:])
        return fixed
    return tgt_turns

# ---------------------------------------------------------------------------
# Pool loading
# ---------------------------------------------------------------------------

def _load_pool(seed: int) -> list[dict[str, Any]]:
    """Stream Dolly_T + OpenAssistant_T and collect only the columns we need.

    Keeps eng_Latn for swap correction, plus the four target language columns.
    Total memory: ~34.9k rows × 5 string fields ≈ well under 50 MB.
    """
    keep_cols = ["eng_Latn"] + [lang for lang, _ in LANG_SLOTS]
    pool: list[dict[str, Any]] = []

    for config in POOL_CONFIGS:
        LOGGER.info("Streaming %s …", config)
        ds = load_dataset(DATASET_ID, config, split="train", streaming=True)
        for row in tqdm(ds, desc=config, unit="rows"):
            pool.append({col: row.get(col) for col in keep_cols})
        gc.collect()

    LOGGER.info("Pool: %d rows from %s", len(pool), " + ".join(POOL_CONFIGS))

    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool


def _validate_pool(pool: list[dict]) -> None:
    needed = sum(count for _, count in LANG_SLOTS)
    if len(pool) < needed:
        raise ValueError(
            f"Pool has only {len(pool)} rows but {needed} are needed "
            f"across all language slots. Adjust LANG_SLOTS counts."
        )

# ---------------------------------------------------------------------------
# Example extraction
# ---------------------------------------------------------------------------

def _extract_indic(pool: list[dict]) -> list[dict[str, str]]:
    """Slice the shuffled pool into disjoint partitions, one per language slot."""
    examples: list[dict[str, str]] = []
    offset = 0

    for lang, count in LANG_SLOTS:
        partition = pool[offset : offset + count]
        extracted = 0
        for record in partition:
            eng_turns = _parse_turns(record.get("eng_Latn"))
            tgt_turns = _parse_turns(record.get(lang))
            tgt_turns = _correct_swap(eng_turns, tgt_turns)
            if tgt_turns:
                prompt, response = tgt_turns[0]
                examples.append({"prompt": prompt, "response": response, "lang": lang})
                extracted += 1
        LOGGER.info("  %s: extracted %d / %d", lang, extracted, count)
        offset += count

    return examples


def _extract_english(seed: int) -> list[dict[str, str]]:
    """Stream Anudesh, take first turn of each conversation, subsample."""
    LOGGER.info("Streaming %s (English) …", ENGLISH_CONFIG)
    ds = load_dataset(DATASET_ID, ENGLISH_CONFIG, split="train", streaming=True)

    rows: list[tuple[str, str]] = []
    for row in tqdm(ds, desc=ENGLISH_CONFIG, unit="rows"):
        turns = _parse_turns(row.get("interactions"))
        if turns:
            rows.append(turns[0])

    gc.collect()
    LOGGER.info("Anudesh: %d usable first turns", len(rows))

    rng = random.Random(seed)
    rng.shuffle(rows)

    examples: list[dict[str, str]] = []
    for prompt, response in rows[:ENGLISH_COUNT]:
        examples.append({"prompt": prompt, "response": response, "lang": "eng_Latn"})
    return examples

# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _write_jsonl(examples: list[dict], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    tmp.replace(path)
    LOGGER.info("Wrote %d examples → %s", len(examples), path)


def _write_manifest(
    out_dir: Path,
    *,
    seed: int,
    lang_counts: dict[str, int],
    n_train: int,
    n_val: int,
) -> None:
    manifest = {
        "sources": {
            "eng_Latn": f"{DATASET_ID} / {ENGLISH_CONFIG}",
            "indic":    f"{DATASET_ID} / {' + '.join(POOL_CONFIGS)} (disjoint row partitions)",
        },
        "seed":        seed,
        "total":       n_train + n_val,
        "n_train":     n_train,
        "n_val":       n_val,
        "val_ratio":   VAL_RATIO,
        "lang_counts": lang_counts,
        "lang_slots":  {lang: count for lang, count in LANG_SLOTS},
        "english_count": ENGLISH_COUNT,
        "note": (
            "Ratios mirror pretraining weights (55/18/13/7/7). "
            "Each source row contributes exactly one language column (disjoint partitions). "
            "Only first turn of each conversation used."
        ),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Manifest → %s", path)

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_sft_data(*, seed: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = _load_pool(seed)
    _validate_pool(pool)

    indic   = _extract_indic(pool)
    del pool
    gc.collect()

    english = _extract_english(seed)

    all_examples = indic + english
    LOGGER.info("Total examples before shuffle: %d", len(all_examples))

    rng = random.Random(seed)
    rng.shuffle(all_examples)

    val_count   = int(len(all_examples) * VAL_RATIO)
    val_ex      = all_examples[:val_count]
    train_ex    = all_examples[val_count:]

    _write_jsonl(train_ex, out_dir / "train.jsonl")
    _write_jsonl(val_ex,   out_dir / "val.jsonl")

    lang_counts: dict[str, int] = {}
    for ex in all_examples:
        lang_counts[ex["lang"]] = lang_counts.get(ex["lang"], 0) + 1

    _write_manifest(
        out_dir,
        seed=seed,
        lang_counts=lang_counts,
        n_train=len(train_ex),
        n_val=len(val_ex),
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    build_sft_data(seed=args.seed, out_dir=args.out_dir)


def _should_force_hard_exit() -> bool:
    raw = os.getenv("MGPT2_FORCE_HARD_EXIT", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


if __name__ == "__main__":
    ok = False
    try:
        main()
        ok = True
    finally:
        logging.shutdown()
        if ok and _should_force_hard_exit():
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
