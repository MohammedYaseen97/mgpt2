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
    git_hash = run_info.get("git_hash", "unknown")[:12]
    train = run_info.get("train", {})

    # ── YAML frontmatter ──────────────────────────────────────────────────────
    base_model_map = {
        "pretrain": "null",
        "sft":      "  # fill in your pretrain repo-id",
        "dpo":      "  # fill in your sft repo-id",
    }
    base_model_line = {
        "pretrain": "",
        "sft":      f"base_model: {repo_id.rsplit('-', 1)[0]}-pretrain\n",
        "dpo":      f"base_model: {repo_id.rsplit('-', 1)[0]}-sft\n",
    }[stage]

    pipeline_tag = "text-generation"
    extra_tags = {
        "pretrain": ["causal-lm", "multilingual", "indic", "hindi", "kannada", "from-scratch"],
        "sft":      ["causal-lm", "multilingual", "indic", "hindi", "kannada", "instruction-tuned", "text-generation-inference"],
        "dpo":      ["causal-lm", "multilingual", "indic", "hindi", "kannada", "instruction-tuned", "dpo", "preference-alignment"],
    }[stage]

    tags_block = "\n".join(f"- {t}" for t in extra_tags)
    frontmatter = f"""---
language:
- en
- hi
- kn
license: mit
tags:
{tags_block}
pipeline_tag: {pipeline_tag}
{base_model_line}---"""

    # ── stage-specific prose ──────────────────────────────────────────────────
    short_name = repo_id.split("/")[-1]

    if stage == "pretrain":
        headline = f"# {short_name} — Multilingual GPT-2 (Pretrained)"
        summary = """\
GPT-2 (124M parameters) trained **from scratch** on a multilingual corpus covering English,
Hindi (Devanagari + Latin transliteration), and Kannada (Kannada script + Latin transliteration).
Trained with a custom BPE tokenizer (`mgpt2`) that achieves 54% better compression than
tiktoken-gpt2 and 38% better than tiktoken-cl100k on the same corpus.

This is the **base pretrained model**. It is a causal language model and will continue text,
not follow instructions. See the SFT and DPO variants for instruction-following versions."""
        use_cases = """\
- Research: study multilingual pretraining dynamics
- Base for fine-tuning on Indic-language tasks
- Tokenizer efficiency benchmarking against tiktoken baselines"""
        not_for = "Direct end-user applications (not instruction-tuned; not safety-filtered)."
        gen_example = f"""\
```python
import sys, torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

local = snapshot_download("{repo_id}")
sys.path.insert(0, local)
from model import GPT
from tokenizer.regex_tokenizer import RegexTokenizer

# Load model
ckpt = torch.load(f"{{local}}/pytorch_model.pt", weights_only=False, map_location="cpu")
model = GPT(ckpt["config"])
model.load_state_dict(ckpt["model"])
model.eval()

# Load tokenizer
enc = RegexTokenizer()
enc.load(f"{{local}}/tokenizer/artifacts/mgpt2.model")

# Generate
prompt = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ"   # "Capital of Karnataka"
ids = enc.encode(prompt)
x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    for _ in range(80):
        logits, _ = model(x[:, -1024:])
        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        if next_id.item() == 50256: break
        x = torch.cat([x, next_id], dim=1)
print(enc.decode(x[0].tolist()))
```"""

    elif stage == "sft":
        headline = f"# {short_name} — Multilingual GPT-2 (Instruction-Tuned)"
        summary = """\
`mgpt2` fine-tuned on **30,000 multilingual instruction–response pairs** across 5 language variants:
English, Hindi (Devanagari), Hindi (Latin transliteration), Kannada (Kannada script), and Kannada
(Latin transliteration). Training data from ai4bharat/indic-align (Anudesh, Dolly-T, OpenAssistant-T).

Built on top of the pretrained `mgpt2` base — same 124M architecture, same custom multilingual tokenizer.
Uses masked cross-entropy (loss computed over response tokens only)."""
        use_cases = """\
- Multilingual Q&A and instruction following (en/hi/kn, native + romanised scripts)
- Downstream fine-tuning starting point for Indic NLP tasks
- Research: multilingual instruction tuning at small scale"""
        not_for = """\
Safety-critical applications. Native-script variants (Devanagari, Kannada) are more reliable than
transliterated Latin variants, which are prone to mid-generation script drift (known limitation —
see training notes)."""
        gen_example = f"""\
```python
import sys, torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

local = snapshot_download("{repo_id}")
sys.path.insert(0, local)
from model import GPT
from tokenizer.regex_tokenizer import RegexTokenizer

ckpt = torch.load(f"{{local}}/pytorch_model.pt", weights_only=False, map_location="cpu")
model = GPT(ckpt["config"])
model.load_state_dict(ckpt["model"])
model.eval()

enc = RegexTokenizer()
enc.load(f"{{local}}/tokenizer/artifacts/mgpt2.model")

# Prompt: plain text, no special template needed
prompts = [
    "What is the capital of Karnataka?",                   # English
    "कर्नाटक की राजधानी क्या है?",                          # Hindi (Devanagari)
    "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?",                           # Kannada script
]

for prompt in prompts:
    ids = enc.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        for _ in range(120):
            logits, _ = model(x[:, -1024:])
            probs = F.softmax(logits[:, -1, :] / 0.7, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if next_id.item() == 50256: break
            x = torch.cat([x, next_id], dim=1)
    print(f"Prompt : {{prompt}}")
    print(f"Response: {{enc.decode(x[0, len(ids):].tolist())}}")
    print()
```"""

    else:  # dpo
        headline = f"# {short_name} — Multilingual GPT-2 (Preference-Aligned)"
        summary = """\
**Recommended model from this project.** `mgpt2-sft` further aligned with
Direct Preference Optimization (DPO, β=0.1) on 13,500 toxic preference pairs
from ai4bharat/indic-align (HHRLHF-T). Chosen responses are Llama2-70B-Chat
safety refusals; rejected responses are raw pretrained-model continuations.

DPO increased the log-probability of safety-refusal responses by **+6.1%** and
decreased the log-probability of rejected responses by **−16.8%** relative to
the SFT checkpoint. Generation comparisons show the SFT model attempts to comply
with harmful prompts; the DPO model redirects. See the [project report](https://github.com)
for full analysis."""
        use_cases = """\
- Multilingual instruction following with light safety alignment (en/hi/kn)
- Research: DPO alignment dynamics at 124M scale
- Demo of end-to-end LLM pipeline: pretrain → SFT → DPO"""
        not_for = """\
Production safety-critical applications. Alignment is format-preference alignment
(coherent refusals vs incoherent noise), not full safety alignment. At 124M
parameters the pretrained model could not generate coherent harmful content, so the
DPO preference signal is weaker than production RLHF setups."""
        gen_example = f"""\
```python
import sys, torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

local = snapshot_download("{repo_id}")
sys.path.insert(0, local)
from model import GPT
from tokenizer.regex_tokenizer import RegexTokenizer

ckpt = torch.load(f"{{local}}/pytorch_model.pt", weights_only=False, map_location="cpu")
model = GPT(ckpt["config"])
model.load_state_dict(ckpt["model"])
model.eval()

enc = RegexTokenizer()
enc.load(f"{{local}}/tokenizer/artifacts/mgpt2.model")

prompts = [
    "Explain what photosynthesis is.",             # English
    "प्रकाश संश्लेषण क्या है?",                    # Hindi (Devanagari)
    "ದ್ಯುತಿಸಂಶ್ಲೇಷಣೆ ಎಂದರೇನು?",                    # Kannada script
]

for prompt in prompts:
    ids = enc.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        for _ in range(120):
            logits, _ = model(x[:, -1024:])
            probs = F.softmax(logits[:, -1, :] / 0.7, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if next_id.item() == 50256: break
            x = torch.cat([x, next_id], dim=1)
    print(f"Prompt : {{prompt}}")
    print(f"Response: {{enc.decode(x[0, len(ids):].tolist())}}")
    print()
```"""

    # ── training details table ────────────────────────────────────────────────
    train_rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in train.items()
                           if k not in ("shards_dir",))

    # ── eval metrics ─────────────────────────────────────────────────────────
    if stage == "pretrain":
        hs = final_metrics.get("final_hellaswag_acc")
        val_loss = final_metrics.get("final_val_loss")
        eval_block = f"""\
| Metric | Value | Notes |
|---|---|---|
| Val loss | {val_loss:.4f} | Cross-entropy on held-out corpus |
| HellaSwag acc | {hs:.4f} | 10,042 examples, own tokenizer |
| BPB overall | 0.809 | bits-per-byte, normalised for tokenizer density |
| BPB vs baseline | −0.071 | mgpt2 better on every language bucket |

> Raw perplexity (12.4) is higher than the GPT-2-tokenized baseline (3.6) — this comparison is **invalid** across tokenizers.
> Bits-per-byte (BPB) is the fair metric and reverses the result: mgpt2 wins on every bucket.
> HellaSwag z=1.59 (directional, not significant at 95% CI); BPB is the primary metric."""
    elif stage == "sft":
        val_loss = final_metrics.get("final_val_loss")
        eval_block = f"""\
| Metric | Value | Notes |
|---|---|---|
| Val loss (masked CE) | {val_loss:.4f} | Response tokens only, held-out SFT set |
| Val PPL (SFT set) | 3.46 | Not comparable to pretrain LM PPL |
| Training steps | {final_metrics.get("total_steps", "—")} | 3 epochs over 30K examples |

> SFT val PPL is measured on the SFT held-out set (narrower domain) and is **not comparable**
> to the pretrain LM eval PPL (12.4), which measures general language modelling ability."""
    else:
        eval_block = f"""\
| Metric | Value | Notes |
|---|---|---|
| Preference win-rate | 1.000 | Held-out DPO pairs (n=1,496) |
| DPO val loss | ~0 | Training converged fully |
| SFT loss regression | +1.2% | Within 5% threshold (regression_ok=True) |
| Chosen log-p Δ | +6.1% | vs SFT checkpoint on same pairs |
| Rejected log-p Δ | −16.8% | vs SFT checkpoint on same pairs |
| Preference margin Δ | +29.1% | chosen − rejected margin widened |

> 100% win-rate reflects format-preference alignment (coherent refusals vs word-salad),
> not full safety alignment. See project report for full generation comparison."""

    # ── training data ─────────────────────────────────────────────────────────
    if stage == "pretrain":
        data_block = """\
| Split | Source | Weight |
|---|---|---|
| English | [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | 55% |
| Hindi (Devanagari) | [AI4Bharat Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) `verified/hin` | 18% |
| Hindi (Latin translit) | AI4Bharat Sangraha `synthetic/hin_Latn` | 7% |
| Kannada (script) | AI4Bharat Sangraha `verified/kan` | 13% |
| Kannada (Latin translit) | AI4Bharat Sangraha `synthetic/kan_Latn` | 7% |

15M documents, globally shuffled, ~40GB raw text. ~14.4B tokens after tokenization (27,537 × 524,288)."""
    elif stage == "sft":
        data_block = """\
| Language | Count | Source |
|---|---|---|
| English (`eng_Latn`) | 16,500 | [ai4bharat/indic-align](https://huggingface.co/datasets/ai4bharat/indic-align) Anudesh |
| Hindi Devanagari (`hin_Deva`) | 5,400 | indic-align Dolly-T + OpenAssistant-T |
| Kannada script (`kan_Knda`) | 3,900 | indic-align Dolly-T + OpenAssistant-T |
| Hindi Latin translit (`hin_Latn`) | 2,100 | indic-align Dolly-T + OpenAssistant-T |
| Kannada Latin translit (`kan_Latn`) | 2,100 | indic-align Dolly-T + OpenAssistant-T |

30,000 examples total. 90/10 train/val split. Masked CE — loss computed over response tokens only."""
    else:
        data_block = """\
| Language | Count | Chosen source | Rejected source |
|---|---|---|---|
| English (`eng_Latn`) | 8,250 | Llama2-70B-Chat safety refusals | Phase C pretrained mgpt2 |
| Hindi Devanagari (`hin_Deva`) | 2,700 | IndicTrans2-translated refusals | Phase C pretrained mgpt2 |
| Kannada script (`kan_Knda`) | 1,950 | IndicTrans2-translated refusals | Phase C pretrained mgpt2 |
| Hindi Latin (`hin_Latn`) | 1,050 | IndicTrans2 romanisation | Phase C pretrained mgpt2 |
| Kannada Latin (`kan_Latn`) | 1,050 | IndicTrans2 romanisation | Phase C pretrained mgpt2 |

13,500 train / 1,499 val pairs. Source: [ai4bharat/indic-align](https://huggingface.co/datasets/ai4bharat/indic-align) HHRLHF-T config."""

    # ── known limitations ─────────────────────────────────────────────────────
    limitations = {
        "pretrain": """\
- **Not instruction-tuned.** The model continues text; it does not follow instructions or answer questions reliably.
- **Transliterated Latin (hin_Latn, kan_Latn)** generation quality is lower than native-script variants — shared ASCII token space with English makes script boundaries ambiguous.
- **124M parameters** — significantly smaller than modern LLMs; factual accuracy and reasoning are limited.
- **Research checkpoint** — not evaluated for safety or production use.""",
        "sft": """\
- **Transliterated Latin script drift.** `hin_Latn` and `kan_Latn` may switch scripts mid-generation. Cause: ASCII tokens shared with English; no Unicode anchor. Mitigated but not eliminated at this data scale.
- **124M parameters.** Factual accuracy and multi-step reasoning are limited.
- **No safety alignment.** The SFT model was trained on benign instruction data only; it may attempt to answer harmful prompts. Use the DPO variant for light safety alignment.
- **Research checkpoint** — not evaluated for production use.""",
        "dpo": """\
- **Format-preference alignment, not full safety alignment.** At 124M parameters, the pretrained model generates incoherent text for toxic prompts, so the DPO preference signal trains format preference (coherent refusals vs noise) rather than genuine safety reasoning.
- **Transliterated Latin script drift** (inherited from SFT checkpoint) — `hin_Latn`/`kan_Latn` may switch scripts mid-generation.
- **124M parameters.** Factual accuracy and multi-step reasoning are limited.
- **Research checkpoint** — not evaluated for production use.""",
    }[stage]

    # ── assemble ──────────────────────────────────────────────────────────────
    content = f"""{frontmatter}

{headline}

{summary}

## Quick start

{gen_example}

## Intended use

**Good for:**
{use_cases}

**Not for:** {not_for}

## Model details

| Property | Value |
|---|---|
| Architecture | GPT-2 (12 layers / 12 heads / 768d) |
| Parameters | ~124M |
| Vocabulary | 50,257 (mgpt2 BPE) + padded to 50,304 |
| Context length | 1,024 tokens |
| Training stage | {STAGE_LABELS[stage]} |
| Git commit | `{git_hash}` |

## Training configuration

| Parameter | Value |
|---|---|
{train_rows}

## Evaluation

{eval_block}

## Training data

{data_block}

## Tokenizer

Custom multilingual regex + BPE tokenizer (`mgpt2`), trained on the same corpus mixture.
Same vocabulary size as tiktoken-gpt2 (50,257 tokens), but with Indic-aware merge priorities:

| Bucket | tiktoken-gpt2 | **mgpt2** | Δ |
|---|---:|---:|---:|
| Overall | 480 tok/kB | **223 tok/kB** | −54% |
| Devanagari | 592 tok/kB | **215 tok/kB** | −64% |
| Kannada | 981 tok/kB | **213 tok/kB** | −78% |
| Latin | 257 tok/kB | **230 tok/kB** | −10% |

Tokenizer published separately: [ace-1/mgpt2-tokenizer](https://huggingface.co/ace-1/mgpt2-tokenizer)

## Known limitations

{limitations}

## Citation

```bibtex
@misc{{mgpt2,
  title     = {{mgpt2: Multilingual GPT-2 with custom Indic tokenizer}},
  year      = {{2026}},
  note      = {{Pretrain → SFT → DPO pipeline for English/Hindi/Kannada}},
  url       = {{https://huggingface.co/{repo_id}}}
}}
```
"""
    (staging / "README.md").write_text(content, encoding="utf-8")


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
