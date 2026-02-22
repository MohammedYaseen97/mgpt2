import argparse
import os
import random
from typing import TextIO


def _reservoir_sample_lines(f: TextIO, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    sample: list[str] = []
    seen = 0
    for raw in f:
        line = raw.rstrip("\n")
        if not line:
            continue
        seen += 1
        if len(sample) < n:
            sample.append(line)
            continue
        j = rng.randrange(seen)
        if j < n:
            sample[j] = line
    if not sample:
        raise SystemExit("No non-empty lines found in corpus.")
    return sample


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a held-out eval set sampled from the training corpus.")
    ap.add_argument("--corpus", required=True, help="Path to corpus text file (one example per line).")
    ap.add_argument("--out", required=True, help="Output text file path.")
    ap.add_argument("--n", type=int, default=5000, help="Number of non-empty lines to sample.")
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed for reproducibility.")
    args = ap.parse_args()

    if args.n <= 0:
        raise SystemExit("--n must be > 0")

    with open(args.corpus, "r", encoding="utf-8") as f:
        sample = _reservoir_sample_lines(f, n=args.n, seed=args.seed)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Shuffle for nicer eval ordering, but deterministically.
    rng = random.Random(args.seed)
    rng.shuffle(sample)

    with open(args.out, "w", encoding="utf-8") as f:
        for s in sample:
            f.write(s)
            f.write("\n")

    print(f"Wrote {len(sample)} lines to {args.out}")


if __name__ == "__main__":
    main()

