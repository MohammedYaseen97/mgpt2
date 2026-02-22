import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def _env_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )


def _copy_minimal_tokenizer_code(dst_root: Path) -> None:
    """
    Copy only the python code required to load `MGPT2Tokenizer` with trust_remote_code.
    Avoid copying huge corpora / notebooks.
    """
    src_root = Path(__file__).resolve().parents[1]  # .../tokenizer
    dst_pkg = dst_root / "tokenizer"
    dst_pkg.mkdir(parents=True, exist_ok=True)

    keep = [
        "__init__.py",
        "base.py",
        "basic.py",
        "regex_tokenizer.py",
        "patterns.py",
        "gpt4.py",
        "hf_tokenizer.py",
    ]
    for name in keep:
        shutil.copy2(src_root / name, dst_pkg / name)


def _patch_tokenizer_config(repo_dir: Path) -> None:
    cfg_path = repo_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Expected {cfg_path} to exist after save_pretrained().")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["tokenizer_class"] = "MGPT2Tokenizer"
    # Module path is relative to repo root when trust_remote_code=True
    cfg["auto_map"] = {"AutoTokenizer": "tokenizer.hf_tokenizer.MGPT2Tokenizer"}
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def _write_repo_readme(repo_dir: Path, repo_id: str, model_path: str, eval_text: str | None, eval_limit: int) -> None:
    lines = []
    lines.append(f"# {repo_id}")
    lines.append("")
    lines.append("Custom mgpt2 tokenizer (pure-Python) exported for Hugging Face `trust_remote_code=True`.")
    lines.append("")
    lines.append("## Usage")
    lines.append("")
    lines.append("```python")
    lines.append("from transformers import AutoTokenizer")
    lines.append("")
    lines.append(f"tok = AutoTokenizer.from_pretrained({repo_id!r}, trust_remote_code=True)")
    lines.append("print(tok.encode('hello world'))")
    lines.append("```")
    lines.append("")
    lines.append("## Contents")
    lines.append(f"- Trained tokenizer artifact: `{Path(model_path).name}` (native `.model` format)")
    lines.append("- Python implementation under `tokenizer/` (loaded via `trust_remote_code`)") 
    lines.append("")
    if eval_text:
        lines.append("## Evaluation")
        lines.append("")
        lines.append(f"Evaluated on `{Path(eval_text).name}` with `--limit {eval_limit}`.")
        lines.append("See `evaluation.json` for metrics (bytes/token, p95 tokens/line, and bucket breakdown).")
        lines.append("")

    (repo_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_evaluation(repo_dir: Path, eval_text: str, eval_limit: int, trained_model: str) -> None:
    # Run the evaluator from *this repo* (not from the staged folder) to keep it fast & consistent.
    cmd = [
        sys.executable,
        "-m",
        "tokenizer.scripts.evaluate",
        "--text",
        eval_text,
        "--limit",
        str(eval_limit),
        "--model",
        trained_model,
    ]
    out = subprocess.check_output(cmd, cwd=Path(__file__).resolve().parents[2])
    (repo_dir / "evaluation.json").write_bytes(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish trained + evaluated mgpt2 tokenizer to Hugging Face Hub.")
    ap.add_argument("--repo_id", required=True, help="HF repo id, e.g. username/mgpt2-tokenizer")
    ap.add_argument("--model", required=True, help="Path to trained .model file (e.g. tokenizer/artifacts/mgpt2.model)")
    ap.add_argument("--private", action="store_true", help="Create/upload as a private repository.")
    ap.add_argument("--eval_text", default=None, help="Optional: path to held-out eval text file.")
    ap.add_argument("--eval_limit", type=int, default=10000, help="How many lines to evaluate (if --eval_text).")
    ap.add_argument("--commit_message", default="Upload mgpt2 tokenizer", help="Hub commit message.")
    ap.add_argument("--dry_run", action="store_true", help="Build staging folder locally but do not upload.")
    ap.add_argument("--staging_dir", default=None, help="Optional: write staging output to this directory (no temp).")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists() or model_path.suffix != ".model":
        raise SystemExit(f"--model must exist and end with .model, got: {args.model}")

    if args.staging_dir:
        repo_dir = Path(args.staging_dir).resolve()
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="mgpt2_tokenizer_hf_")
        repo_dir = Path(temp_ctx.name)

    try:
        # 1) Save tokenizer in HF format into repo_dir
        from tokenizer.hf_tokenizer import MGPT2Tokenizer

        tok = MGPT2Tokenizer(model_file=str(model_path))
        # Don't pass filename_prefix here; HF expects standard filenames like tokenizer_config.json
        tok.save_pretrained(str(repo_dir))

        # 2) Add minimal python code required for trust_remote_code
        _copy_minimal_tokenizer_code(repo_dir)
        _patch_tokenizer_config(repo_dir)

        # 3) Optionally evaluate and attach results
        if args.eval_text:
            _run_evaluation(repo_dir, args.eval_text, args.eval_limit, str(model_path))

        # 4) Write README
        _write_repo_readme(repo_dir, args.repo_id, str(model_path), args.eval_text, args.eval_limit)

        if args.dry_run:
            print(f"[dry_run] Staged files at: {repo_dir}")
            return

        token = _env_token()
        if not token:
            raise SystemExit("Missing HF token. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in your environment.")

        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(repo_dir),
            commit_message=args.commit_message,
        )
        print(f"Uploaded to {args.repo_id}")
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    main()

