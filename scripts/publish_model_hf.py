"""
Publish a trained mgpt2 checkpoint to Hugging Face Hub.

Supports all three training stages:
  --stage pretrain   runs/baseline_*/  or  runs/mgpt2_*/
  --stage sft        runs/sft_*/
  --stage dpo        runs/dpo_*/

Usage:
  python scripts/publish_model_hf.py \\
      --stage pretrain \\
      --run-dir runs/mgpt2_custom_tokenizer_smoke_... \\
      --repo-id yourname/mgpt2-pretrain \\
      [--private] [--dry-run]

Requires HF_TOKEN env variable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

STAGE_LABELS = {
    "pretrain": "Pretrained",
    "sft": "SFT (instruction-tuned)",
    "dpo": "DPO (preference-aligned)",
}

STAGE_DESCRIPTIONS = {
    "pretrain": (
        "Causal language model trained from scratch on a multilingual corpus "
        "(English + Hindi + Kannada, Devanagari + Latin-script)."
    ),
    "sft": (
        "Instruction-tuned on IndicAlign (Dolly-T / OpenAssistant-T / Anudesh); "
        "fine-tuned on top of the pretrained mgpt2 checkpoint."
    ),
    "dpo": (
        "Preference-aligned with DPO on IndicAlign HHRLHF-T + Toxic-Matrix; "
        "fine-tuned on top of the SFT checkpoint."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _env_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )


def _latest_run(prefix: str) -> Path | None:
    candidates = sorted(
        [d for d in (REPO_ROOT / "runs").iterdir() if d.is_dir() and d.name.startswith(prefix)]
    )
    return candidates[-1] if candidates else None


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.glob("model_*.pt"))
    if not checkpoints:
        raise SystemExit(f"No model_*.pt checkpoint found in {run_dir}")
    return checkpoints[-1]


def _copy_tokenizer_code(staging: Path) -> None:
    """Bundle the tokenizer package needed for trust_remote_code loading."""
    src = REPO_ROOT / "tokenizer"
    dst = staging / "tokenizer"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "base.py", "regex_tokenizer.py", "patterns.py"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)

    # Stable root-level entrypoint so transformers can do:
    #   AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    (staging / "tokenization_mgpt2.py").write_text(
        "from tokenizer.hf_tokenizer import MGPT2Tokenizer\n\n"
        "__all__ = ['MGPT2Tokenizer']\n",
        encoding="utf-8",
    )
    hf_tok_src = src / "hf_tokenizer.py"
    if hf_tok_src.exists():
        shutil.copy2(hf_tok_src, dst / "hf_tokenizer.py")


def _write_config(staging: Path, run_info: dict) -> None:
    """Write config.json understood by trust_remote_code model loading."""
    import sys
    import torch
    sys.path.insert(0, str(REPO_ROOT))
    from model import GPTConfig

    ckpt = torch.load(staging / "pytorch_model.pt", weights_only=False, map_location="cpu")
    cfg: GPTConfig = ckpt["config"]
    tok_kind = run_info.get("tokenizer", {}).get("kind", "gpt2_tiktoken")

    config_dict = {
        "architectures": ["GPT"],
        "model_type": "mgpt2",
        **{k: v for k, v in asdict(cfg).items()},
        "tokenizer_kind": tok_kind,
    }
    (staging / "config.json").write_text(
        json.dumps(config_dict, indent=2) + "\n", encoding="utf-8"
    )


def _write_readme(
    staging: Path,
    repo_id: str,
    stage: str,
    run_info: dict,
    final_metrics: dict,
) -> None:
    tok = run_info.get("tokenizer", {})
    tok_kind = tok.get("kind", "gpt2_tiktoken")
    tok_label = "GPT-2 tiktoken (baseline)" if tok_kind == "gpt2_tiktoken" else "mgpt2 custom BPE"
    git_hash = run_info.get("git_hash", "unknown")
    train = run_info.get("train", {})
    stage_label = STAGE_LABELS[stage]

    hs_acc = final_metrics.get("final_hellaswag_acc")
    hs_str = f"{hs_acc:.4f}" if hs_acc is not None else "—"
    val_loss = final_metrics.get("final_val_loss")
    val_str = f"{val_loss:.4f}" if val_loss is not None else "—"

    load_snippet = f"""```python
import torch
from huggingface_hub import hf_hub_download
import sys, os

# Download model files
ckpt_path  = hf_hub_download("{repo_id}", "pytorch_model.pt")
model_path = hf_hub_download("{repo_id}", "model.py")

sys.path.insert(0, os.path.dirname(model_path))
from model import GPT

checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")
model = GPT(checkpoint["config"])
model.load_state_dict(checkpoint["model"])
model.eval()
print(f"Loaded step {{checkpoint['step']}}, val_loss={{checkpoint['val_loss']:.4f}}")
```"""

    if tok_kind != "gpt2_tiktoken":
        tok_snippet = f"""```python
from huggingface_hub import snapshot_download
from tokenizer.regex_tokenizer import RegexTokenizer

local = snapshot_download("{repo_id}")
enc = RegexTokenizer()
enc.load(f"{{local}}/tokenizer/artifacts/mgpt2.model")
ids = enc.encode("नमस्ते! Hello!")
print(enc.decode(ids))
```"""
    else:
        tok_snippet = """```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Hello world!")
print(enc.decode(ids))
```"""

    lines = [
        f"# {repo_id}",
        "",
        f"**Stage**: {stage_label}  ",
        f"**Tokenizer**: {tok_label}  ",
        f"**Git commit**: `{git_hash[:12]}`",
        "",
        STAGE_DESCRIPTIONS[stage],
        "",
        "## Training details",
        "",
        "| Parameter | Value |",
        "|---|---|",
        *[f"| `{k}` | `{v}` |" for k, v in train.items()],
        "",
        "## Evaluation",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Val loss | {val_str} |",
        f"| HellaSwag acc (inline) | {hs_str} |",
        "",
        "> HellaSwag evaluated with this model's own tokenizer.",
        "> Contextual HF GPT-2 comparison (different training data) is reported separately.",
        "",
        "## Corpus",
        "",
        "Trained on a fixed multilingual mixture:",
        "- 55% FineWeb (English)",
        "- 18% Sangraha `verified/hin` (Devanagari Hindi)",
        "- 7%  Sangraha `synthetic/hin_Latn` (transliterated Hindi)",
        "- 13% Sangraha `verified/kan` (Kannada script)",
        "- 7%  Sangraha `synthetic/kan_Latn` (transliterated Kannada)",
        "",
        "## Loading",
        "",
        load_snippet,
        "",
        "### Tokenizer",
        "",
        tok_snippet,
        "",
        "## Notes",
        "",
        "- Architecture: Karpathy-style GPT-2 (12L / 12H / 768d, ~124M params)",
        "- Vocab size: 50,257 (GPT-2 exact terms); embedding matrix padded to 50,304",
        "- This is a **research checkpoint** — not safety-evaluated for production use",
        "",
    ]
    (staging / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Publish an mgpt2 checkpoint to Hugging Face Hub.")
    ap.add_argument(
        "--stage",
        required=True,
        choices=["pretrain", "sft", "dpo"],
        help="Training stage of this checkpoint.",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to run directory (default: latest matching runs/ dir for the stage).",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit path to a .pt checkpoint (overrides --run-dir checkpoint discovery).",
    )
    ap.add_argument("--repo-id", required=True, help="HF repo id, e.g. yourname/mgpt2-pretrain")
    ap.add_argument("--private", action="store_true", help="Create as a private repository.")
    ap.add_argument("--dry-run", action="store_true", help="Stage files locally, skip upload.")
    ap.add_argument(
        "--commit-message",
        default=None,
        help="Hub commit message (auto-generated if omitted).",
    )
    ap.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Write staged files here instead of a temp directory (useful for inspection).",
    )
    args = ap.parse_args()

    # ── resolve run dir ───────────────────────────────────────────────────────
    prefix_map = {"pretrain": ("baseline_", "mgpt2_"), "sft": ("sft_",), "dpo": ("dpo_",)}
    if args.run_dir is None:
        for prefix in prefix_map[args.stage]:
            args.run_dir = _latest_run(prefix)
            if args.run_dir:
                break
        if args.run_dir is None:
            raise SystemExit(f"No run directory found for stage '{args.stage}'.")
    if not args.run_dir.exists():
        raise SystemExit(f"Run directory not found: {args.run_dir}")

    run_info_path = args.run_dir / "run_info.json"
    if not run_info_path.exists():
        raise SystemExit(f"Missing run_info.json in {args.run_dir}")
    run_info = json.loads(run_info_path.read_text())

    final_metrics_path = args.run_dir / "final_metrics.json"
    final_metrics = json.loads(final_metrics_path.read_text()) if final_metrics_path.exists() else {}

    checkpoint = args.checkpoint or _latest_checkpoint(args.run_dir)
    print(f"  stage      : {args.stage}")
    print(f"  run dir    : {args.run_dir.name}")
    print(f"  checkpoint : {checkpoint.name}")
    print(f"  repo       : {args.repo_id}")

    # ── stage files ───────────────────────────────────────────────────────────
    if args.staging_dir:
        staging = args.staging_dir.resolve()
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="mgpt2_hf_")
        staging = Path(temp_ctx.name)

    try:
        # 1. checkpoint
        shutil.copy2(checkpoint, staging / "pytorch_model.pt")

        # 2. model architecture
        shutil.copy2(REPO_ROOT / "model.py", staging / "model.py")

        # 3. config.json (loads the checkpoint to read GPTConfig)
        _write_config(staging, run_info)

        # 4. tokenizer artifacts (mgpt2 only)
        tok_kind = run_info.get("tokenizer", {}).get("kind", "gpt2_tiktoken")
        if tok_kind != "gpt2_tiktoken":
            tok_model_src = Path(run_info["tokenizer"].get("path", "tokenizer/artifacts/mgpt2.model"))
            if not tok_model_src.is_absolute():
                tok_model_src = REPO_ROOT / tok_model_src
            dst_tok_artifacts = staging / "tokenizer" / "artifacts"
            dst_tok_artifacts.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tok_model_src, dst_tok_artifacts / tok_model_src.name)
            _copy_tokenizer_code(staging)

        # 5. README
        _write_readme(staging, args.repo_id, args.stage, run_info, final_metrics)

        print(f"\n  staged files:")
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                size = f.stat().st_size
                print(f"    {f.relative_to(staging)}  ({size:,} bytes)")

        if args.dry_run:
            print(f"\n[dry-run] skipping upload. staged at: {staging}")
            if temp_ctx:
                input("press Enter to clean up temp dir ...")
            return

        # ── upload ────────────────────────────────────────────────────────────
        token = _env_token()
        if not token:
            raise SystemExit("Missing HF token. Set HF_TOKEN in your environment.")

        from huggingface_hub import HfApi

        commit_msg = args.commit_message or (
            f"Publish mgpt2 {args.stage} checkpoint "
            f"(step {final_metrics.get('total_steps', '?')}, "
            f"val_loss {final_metrics.get('final_val_loss', '?')})"
        )

        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(staging),
            commit_message=commit_msg,
        )
        print(f"\nUploaded to https://huggingface.co/{args.repo_id}")

    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    main()
