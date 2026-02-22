import argparse
import os

from tokenizer.hf_tokenizer import MGPT2Tokenizer


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
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

