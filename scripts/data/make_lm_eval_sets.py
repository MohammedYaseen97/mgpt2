"""Cut a fixed eval slice from corpus_mixture.txt and bucket by script.

Skipping large offsets:
  When --eval-start is non-zero (e.g. 14_850_000), Python islice must
  allocate and discard a Python string object per skipped line — roughly
  15 million objects for a 15M-line file — inflating the heap until the
  OOM-killer arrives.

  This script avoids that by delegating the skip to the shell:
    tail -n +{eval_start+1} | head -n {eval_size}
  The scan runs entirely in C at full disk speed with no Python heap cost.
  When eval_start == 0 the file is opened directly (no subprocess overhead).

  tokenize_shards.py reads the manifest produced here and skips
  eval_line_count lines from eval_line_start so there is no train/eval overlap.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import IO

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS_FILE = REPO_ROOT / "data" / "raw" / "corpus_mixture.txt"
DEFAULT_EVAL_DIR    = REPO_ROOT / "data" / "eval"

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut an eval slice from corpus_mixture.txt and split by script bucket.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus-file", type=Path, default=DEFAULT_CORPUS_FILE)
    parser.add_argument("--eval-dir",    type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--eval-start",  type=int,  default=0,
                        help="Line offset to start the eval slice (0-indexed). "
                             "Large offsets are skipped at the OS level via tail, "
                             "not in Python, so RAM usage is unaffected.")
    parser.add_argument("--eval-size",   type=int,  default=50_000,
                        help="Number of lines in the eval slice.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Script bucketing
# ---------------------------------------------------------------------------

# Compiled patterns — much faster than a per-character Python loop.
_RE_DEVA  = re.compile(r'[\u0900-\u097F]')
_RE_KNDA  = re.compile(r'[\u0C80-\u0CFF]')
_RE_SPACE = re.compile(r'\s')

_THRESHOLD = 0.85


def _get_script_bucket(text: str) -> str:
    """Return the dominant script bucket for a single document line."""
    non_space = _RE_SPACE.sub('', text)
    total = len(non_space)
    if total == 0:
        return "mixed"

    n_deva = len(_RE_DEVA.findall(non_space))
    if n_deva / total >= _THRESHOLD:
        return "deva"

    n_knda = len(_RE_KNDA.findall(non_space))
    if n_knda / total >= _THRESHOLD:
        return "knda"

    # Latin: chars in ASCII range (0x00–0x7F) — count what's left after
    # removing Devanagari, Kannada, and everything non-ASCII.
    n_non_ascii = sum(1 for c in non_space if ord(c) > 0x007F)
    n_latin = total - n_non_ascii
    if n_latin / total >= _THRESHOLD:
        return "latin"

    return "mixed"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _open_eval_stream(
    corpus_file: Path,
    eval_start: int,
    eval_size: int,
) -> tuple[IO[str], subprocess.Popen | None]:
    """Return (line_iterator, optional_proc) for the requested eval slice.

    eval_start == 0: open the file directly — no subprocess.
    eval_start  > 0: use `tail -n +N | head -n M` to skip at the OS level.
                     Python never allocates a string object for skipped lines.
    The caller must close the stream and (if present) wait on the process.
    """
    if eval_start == 0:
        return corpus_file.open("r", encoding="utf-8"), None

    # tail -n +N is 1-indexed: "+1" is the first line, so "+eval_start+1" skips
    # exactly eval_start lines before streaming the rest.
    proc = subprocess.Popen(
        f"tail -n +{eval_start + 1} {corpus_file} | head -n {eval_size}",
        shell=True,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return proc.stdout, proc  # type: ignore[return-value]


def _make_eval_buckets(
    corpus_file: Path,
    eval_dir: Path,
    eval_start: int,
    eval_size: int,
) -> dict[str, dict]:
    """Stream eval_size lines starting at eval_start and write per-bucket files."""
    eval_dir.mkdir(parents=True, exist_ok=True)

    bucket_names  = ["latin", "deva", "knda", "mixed"]
    bucket_fps    = {s: (eval_dir / f"eval_{s}.txt").open("w", encoding="utf-8")
                     for s in bucket_names}
    bucket_counts = {s: 0 for s in bucket_names}

    stream, proc = _open_eval_stream(corpus_file, eval_start, eval_size)
    try:
        for i, line in enumerate(tqdm(stream, total=eval_size,
                                      desc="Bucketing eval lines", unit="lines")):
            if i >= eval_size:
                break
            bucket = _get_script_bucket(line)
            bucket_fps[bucket].write(line)
            bucket_counts[bucket] += 1
    finally:
        stream.close()
        if proc is not None:
            proc.wait()
        for fp in bucket_fps.values():
            fp.close()

    return {
        s: {"file": str(eval_dir / f"eval_{s}.txt"), "line_count": bucket_counts[s]}
        for s in bucket_names
    }


def _write_manifest(
    eval_dir: Path,
    corpus_file: Path,
    eval_start: int,
    eval_size: int,
    buckets: dict[str, dict],
) -> Path:
    total = sum(b["line_count"] for b in buckets.values())
    if total != eval_size:
        raise RuntimeError(f"Bucket line counts sum to {total}, expected {eval_size}.")

    manifest = {
        "source_file":      str(corpus_file),
        "eval_line_start":  eval_start,
        "eval_line_end":    eval_start + eval_size,
        "eval_line_count":  eval_size,
        "buckets":          buckets,
    }
    path = eval_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    LOGGER.info("Cutting %d eval lines from %s (start=%d)",
                args.eval_size, args.corpus_file, args.eval_start)
    buckets = _make_eval_buckets(args.corpus_file, args.eval_dir,
                                 args.eval_start, args.eval_size)

    manifest_path = _write_manifest(args.eval_dir, args.corpus_file,
                                    args.eval_start, args.eval_size, buckets)
    LOGGER.info("Wrote manifest to %s", manifest_path)

    for name, info in buckets.items():
        LOGGER.info("  %-8s  %6d lines", name, info["line_count"])


if __name__ == "__main__":
    main()
