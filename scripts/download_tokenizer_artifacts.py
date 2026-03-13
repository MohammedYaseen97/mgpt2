"""Download tokenizer artifacts from a HuggingFace repo and place them
into tokenizer/artifacts/ with the canonical local filenames.

HF repo layout        →  local artifacts/
  tokenizer.model     →  mgpt2.model
  tokenizer.vocab     →  mgpt2.vocab          (optional, may not exist)
  evaluation.json     →  tokenizer_eval.json  (optional)
  heldout_eval.txt    →  heldout_eval.txt     (optional)

Usage:
  python scripts/download_tokenizer_artifacts.py --repo_id ace-1/mgpt2-tokenizer
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "tokenizer" / "artifacts"

# (hf_filename, local_filename, required)
FILE_MAP = [
    ("tokenizer.model", "mgpt2.model", True),
    ("tokenizer.vocab", "mgpt2.vocab", True),
    ("evaluation.json", "tokenizer_eval.json", True),
    ("heldout_eval.txt", "heldout_eval.txt", True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help="HF repo id, e.g. ace-1/mgpt2-tokenizer")
    ap.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = ap.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for hf_name, local_name, required in FILE_MAP:
        try:
            cached = hf_hub_download(
                repo_id=args.repo_id,
                filename=hf_name,
                token=args.token,
            )
            dest = ARTIFACTS_DIR / local_name
            shutil.copy2(cached, dest)
            print(f"  {hf_name} → {dest.relative_to(REPO_ROOT)}")
        except Exception as e:
            if required:
                raise SystemExit(f"Failed to download required file '{hf_name}': {e}") from e
            print(f"  {hf_name} not found in repo (optional, skipping)")

    print(f"\nDone. Artifacts at: {ARTIFACTS_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
