import argparse
import json
import os
from pathlib import Path

from tokenizer.hf_tokenizer import MGPT2Tokenizer


def _patch_tokenizer_config(out_dir: str) -> None:
    cfg_path = Path(out_dir) / "tokenizer_config.json"
    if not cfg_path.exists():
        # If save_pretrained didn't create it, nothing we can do.
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["tokenizer_class"] = "MGPT2Tokenizer"
    # transformers==5.x expects a 2-item list: [slow_ref, fast_ref]
    cfg["auto_map"] = {"AutoTokenizer": ["tokenization_mgpt2.MGPT2Tokenizer", None]}
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a trained mgpt2 tokenizer in HF format (trust_remote_code).")
    ap.add_argument("--model", required=True, help="Path to trained .model file (e.g. tokenizer/mgpt2.model).")
    ap.add_argument("--out", required=True, help="Output directory (will be created).")
    ap.add_argument(
        "--prefix",
        default=None,
        help="Optional filename prefix (not recommended for HF; may rename tokenizer_config files).",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = MGPT2Tokenizer(model_file=args.model)
    if args.prefix:
        tok.save_pretrained(args.out, filename_prefix=args.prefix)
    else:
        tok.save_pretrained(args.out)
    _patch_tokenizer_config(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

