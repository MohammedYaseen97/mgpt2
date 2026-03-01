"""Build a shuffled text corpus by mixing streamed Hugging Face datasets.

Corpus sources:
  - FineWeb (English web text)
  - Sangraha synthetic/hin_Deva  (Hindi, Devanagari script)
  - Sangraha synthetic/hin_Latn  (Hindi, transliterated Latin)
  - Sangraha synthetic/kan_Knda  (Kannada, Kannada script)
  - Sangraha synthetic/kan_Latn  (Kannada, transliterated Latin)

Outputs:
  - <output-file>     line-based corpus (one document per line)
  - manifest.json     dataset identifiers + sampling parameters for reproducibility
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Iterable

from datasets import interleave_datasets, load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_FILE = REPO_ROOT / "data" / "raw" / "corpus_mixture.txt"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LIMIT = 1_500_000       # ~1B tokens at ~650 tokens/doc average
DEFAULT_SEED = 42
DEFAULT_BUFFER_SIZE = 100_000   # Larger buffer = better shuffle quality

# Sampling weights — must sum to 1.0.
# Native-script sources weighted slightly above transliterated counterparts
# so the tokenizer's non-Latin merges are well represented.
DEFAULT_FINEWEB_WEIGHT          = 0.60
DEFAULT_SANGRAHA_HI_DEVA_WEIGHT = 0.12   # Hindi, Devanagari
DEFAULT_SANGRAHA_HI_LATN_WEIGHT = 0.08   # Hindi, transliterated
DEFAULT_SANGRAHA_KAN_KNDA_WEIGHT = 0.12  # Kannada, native script
DEFAULT_SANGRAHA_KAN_LATN_WEIGHT = 0.08  # Kannada, transliterated

# ---------------------------------------------------------------------------
# Dataset registry — single source of truth for HF dataset IDs / data_dirs
# ---------------------------------------------------------------------------

DATASETS = {
    "fineweb":            {"path": "HuggingFaceFW/fineweb", "name": "sample-10BT"},
    "sangraha_hin_deva":  {"path": "ai4bharat/sangraha", "data_dir": "synthetic/hin_Deva"},
    "sangraha_hin_latn":  {"path": "ai4bharat/sangraha", "data_dir": "synthetic/hin_Latn"},
    "sangraha_kan_knda":  {"path": "ai4bharat/sangraha", "data_dir": "synthetic/kan_Knda"},
    "sangraha_kan_latn":  {"path": "ai4bharat/sangraha", "data_dir": "synthetic/kan_Latn"},
}

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a pretraining corpus mixture from FineWeb and Sangraha (streaming).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    parser.add_argument("--buffer-size", type=int,   default=DEFAULT_BUFFER_SIZE,
                        help="Shuffle buffer size (higher = better quality, more RAM).")
    parser.add_argument("--limit",       type=int,   default=DEFAULT_LIMIT,
                        help="Number of non-empty documents to write.")
    parser.add_argument("--output-file", type=Path,  default=DEFAULT_OUTPUT_FILE)

    # Per-source sampling weights — must sum to 1.0
    parser.add_argument("--fineweb-weight",           type=float, default=DEFAULT_FINEWEB_WEIGHT)
    parser.add_argument("--sangraha-hi-deva-weight",  type=float, default=DEFAULT_SANGRAHA_HI_DEVA_WEIGHT)
    parser.add_argument("--sangraha-hi-latn-weight",  type=float, default=DEFAULT_SANGRAHA_HI_LATN_WEIGHT)
    parser.add_argument("--sangraha-kan-knda-weight", type=float, default=DEFAULT_SANGRAHA_KAN_KNDA_WEIGHT)
    parser.add_argument("--sangraha-kan-latn-weight", type=float, default=DEFAULT_SANGRAHA_KAN_LATN_WEIGHT)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inputs(
    limit: int,
    buffer_size: int,
    weights: dict[str, float],
) -> None:
    """Validate runtime inputs early with clear error messages."""
    if limit <= 0:
        raise ValueError("--limit must be > 0.")
    if buffer_size <= 0:
        raise ValueError("--buffer-size must be > 0.")

    for name, w in weights.items():
        if w <= 0:
            raise ValueError(f"Weight for '{name}' must be positive, got {w}.")

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Sampling weights must sum to 1.0, got {total:.8f}. "
            f"Weights: {weights}"
        )


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

def normalize_text(example: dict[str, Any]) -> str:
    """Extract and normalise the text field of a dataset example."""
    return str(example.get("text", "")).strip().replace("\n", " ")


def _safe_close(iterator: Any) -> None:
    """Best-effort iterator shutdown to reduce background-thread teardown issues."""
    close_fn = getattr(iterator, "close", None)
    if callable(close_fn):
        close_fn()


def _load_stream(key: str) -> Any:
    """Load a single streaming dataset by its registry key."""
    cfg = DATASETS[key]
    kwargs: dict[str, Any] = {"split": "train", "streaming": True}
    if "name" in cfg:
        kwargs["name"] = cfg["name"]
    if "data_dir" in cfg:
        kwargs["data_dir"] = cfg["data_dir"]
    return load_dataset(cfg["path"], **kwargs)


def _iter_mixed_stream(
    seed: int,
    buffer_size: int,
    weights: dict[str, float],
) -> Iterable[dict[str, Any]]:
    """Create a deterministically shuffled interleaved stream from all sources."""
    streams      = [_load_stream(k) for k in weights]
    probabilities = list(weights.values())

    mixed = interleave_datasets(streams, probabilities=probabilities, seed=seed)
    return mixed.shuffle(seed=seed, buffer_size=buffer_size)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(
    output_file: Path,
    seed: int,
    buffer_size: int,
    limit: int,
    weights: dict[str, float],
    lines_written: int,
) -> None:
    manifest = {
        "corpus_file": str(output_file),
        "lines_written": lines_written,
        "seed": seed,
        "buffer_size": buffer_size,
        "limit": limit,
        "datasets": DATASETS,
        "weights": weights,
    }
    manifest_path = output_file.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote manifest to %s", manifest_path)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_corpus_mixture(
    *,
    seed: int,
    buffer_size: int,
    limit: int,
    output_file: Path,
    weights: dict[str, float],
) -> int:
    """Build and persist a mixed pretraining corpus.

    Returns:
        Number of document lines written.
    """
    validate_inputs(limit, buffer_size, weights)

    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")

    LOGGER.info("Source weights: %s", weights)
    LOGGER.info("Preparing streamed dataset mixture (limit=%d, buffer=%d, seed=%d)...",
                limit, buffer_size, seed)

    shuffled_stream = _iter_mixed_stream(seed=seed, buffer_size=buffer_size, weights=weights)
    iterator = iter(shuffled_stream)

    written = 0
    try:
        with temp_file.open("w", encoding="utf-8") as f, tqdm(
            total=limit,
            desc="Writing corpus",
            unit="docs",
        ) as pbar:
            while written < limit:
                example = next(iterator)
                text = normalize_text(example)
                if not text:
                    continue
                f.write(text + "\n")
                written += 1
                pbar.update(1)
    finally:
        _safe_close(iterator)
        gc.collect()

    temp_file.replace(output_file)
    LOGGER.info("Wrote %d documents to %s", written, output_file)

    _write_manifest(output_file, seed, buffer_size, limit, weights, written)
    return written


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    weights = {
        "fineweb":            args.fineweb_weight,
        "sangraha_hin_deva":  args.sangraha_hi_deva_weight,
        "sangraha_hin_latn":  args.sangraha_hi_latn_weight,
        "sangraha_kan_knda":  args.sangraha_kan_knda_weight,
        "sangraha_kan_latn":  args.sangraha_kan_latn_weight,
    }

    build_corpus_mixture(
        seed=args.seed,
        buffer_size=args.buffer_size,
        limit=args.limit,
        output_file=args.output_file,
        weights=weights,
    )


def _should_force_hard_exit() -> bool:
    """Return whether to bypass interpreter teardown after successful execution."""
    raw_value = os.getenv("MGPT2_FORCE_HARD_EXIT", "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


if __name__ == "__main__":
    ok = False
    try:
        main()
        ok = True
    finally:
        logging.shutdown()

        # Workaround for intermittent native-extension teardown crashes observed
        # after successful streamed dataset writes (PyArrow/HF stack on WSL).
        if ok and _should_force_hard_exit():
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
