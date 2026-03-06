"""Test the awk-based external shuffle pipeline.

The awk | sort | cut pipeline is the recommended way to shuffle corpus_mixture.txt
(15M lines, ~15 GB) because:
  - awk prefixes each line with rand() in O(n) streaming — no RAM accumulation.
  - GNU sort spills to disk automatically when input exceeds --sort-buffer, so
    even a 15 GB file fits within a modest RAM budget.
  - cut strips the prefix back off, leaving a plain shuffled text file.
  - The seed is forwarded to awk's srand() so the shuffle is reproducible.

Run against the dummy file first to verify correctness, then point --input at
corpus_mixture.txt when you're confident.

Usage examples:
    # Sanity-check on dummy file (fast, safe, default)
    python scripts/data/test_shuffle.py --seed 42

    # Full corpus (long-running — make sure you have ~2× the file size in free disk)
    python scripts/data/test_shuffle.py \\
        --seed 42 \\
        --input  data/raw/corpus_mixture.txt \\
        --output data/raw/corpus_mixture_shuffled.txt \\
        --sort-buffer 4G
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT  = REPO_ROOT / "data" / "raw" / "dummy_input.txt"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "shuffled_dummy_input.txt"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shuffle a large text file using awk + GNU sort (disk-safe, reproducible).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed",         type=int,  default=42,
                        help="Integer seed forwarded to awk srand().")
    parser.add_argument("--input",        type=Path, default=DEFAULT_INPUT,
                        help="Path to the file to shuffle.")
    parser.add_argument("--output",       type=Path, default=DEFAULT_OUTPUT,
                        help="Path to write the shuffled output.")
    parser.add_argument("--sort-buffer",  type=str,  default="2G",
                        help="Memory budget passed to GNU sort -S (e.g. 2G, 4G). "
                             "sort spills to disk beyond this, so set it safely below "
                             "your free RAM rather than as large as possible.")
    parser.add_argument("--preview-lines", type=int, default=5,
                        help="Number of tail lines to display when verifying shuffle quality.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_lines(path: Path) -> int:
    """Count lines using wc -l (streams the file, no RAM accumulation)."""
    result = subprocess.run(
        ["wc", "-l", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.split()[0])


def tail_lines(path: Path, n: int) -> list[str]:
    """Return the last n lines of a file without loading the whole file."""
    result = subprocess.run(
        ["tail", "-n", str(n), str(path)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.splitlines()


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def shuffle_file(
    input_path: Path,
    output_path: Path,
    seed: int,
    sort_buffer: str,
) -> None:
    """Shuffle input_path → output_path using awk | sort | cut.

    The pipeline never loads the full file into Python memory:
      1. awk  — prepends rand() to every line (streaming)
      2. sort — sorts by the random prefix (disk-spill safe via -S budget)
      3. cut  — strips the random prefix, leaving shuffled plain text
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    LOGGER.info("Shuffling %s → %s (seed=%d, sort-buffer=%s)",
                input_path, output_path, seed, sort_buffer)

    awk_cmd = f"awk -v seed={seed} 'BEGIN {{srand(seed)}} {{print rand(), $0}}'"
    sort_cmd = f"sort -n -S {sort_buffer}"
    cut_cmd  = "cut -d ' ' -f2-"
    pipeline  = f"{awk_cmd} {input_path!s} | {sort_cmd} | {cut_cmd} > {temp_path!s}"

    result = subprocess.run(pipeline, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        LOGGER.error("Pipeline stderr:\n%s", result.stderr)
        raise RuntimeError(f"Shuffle pipeline failed with exit code {result.returncode}.")

    temp_path.rename(output_path)
    LOGGER.info("Shuffle complete → %s", output_path)


def verify_shuffle(
    input_path: Path,
    output_path: Path,
    preview_lines: int,
) -> bool:
    """Verify shuffle correctness; return True if everything checks out."""
    LOGGER.info("Verifying shuffle...")

    n_in  = count_lines(input_path)
    n_out = count_lines(output_path)

    print(f"\n{'─' * 60}")
    print(f"  Input  lines : {n_in:,}")
    print(f"  Output lines : {n_out:,}")

    if n_in != n_out:
        LOGGER.error("LINE COUNT MISMATCH — input %d vs output %d", n_in, n_out)
        return False
    print("  Line count   : ✓ match")

    tail_in  = tail_lines(input_path,  preview_lines)
    tail_out = tail_lines(output_path, preview_lines)

    print(f"\n  Last {preview_lines} lines of INPUT (should be in original order):")
    for line in tail_in:
        print(f"    {line}")

    print(f"\n  Last {preview_lines} lines of OUTPUT (should be shuffled):")
    for line in tail_out:
        print(f"    {line}")

    order_changed = tail_in != tail_out
    if order_changed:
        print("\n  Order check  : ✓ tail differs — shuffle is in effect")
    else:
        print("\n  Order check  : ⚠ tail is identical — "
              "shuffle may not have changed the order (unlikely but possible with small files)")

    print(f"{'─' * 60}\n")
    return True


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    input_path  = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        LOGGER.error("Input file not found: %s", input_path)
        sys.exit(1)

    if output_path == input_path:
        LOGGER.error("--output must differ from --input to avoid overwriting the source.")
        sys.exit(1)

    LOGGER.info("Input  : %s", input_path)
    LOGGER.info("Output : %s", output_path)

    shuffle_file(input_path, output_path, seed=args.seed, sort_buffer=args.sort_buffer)
    ok = verify_shuffle(input_path, output_path, preview_lines=args.preview_lines)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
