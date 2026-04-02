"""SFT evaluation — val masked loss + fixed multilingual prompt suite.

Usage:
    python -m eval.sft_eval \\
        --checkpoint runs/<sft_run>/model_NNNNN.pt \\
        --tokenizer-model tokenizer/artifacts/mgpt2.model \\
        --shards-dir data/shards_sft \\
        --out runs/<sft_run>/sft_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import types as _types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

EOT_TOKEN  = 50256
BLOCK_SIZE = 1024
EVAL_BATCH = 8

# ---------------------------------------------------------------------------
# Fixed multilingual prompt suite (2 per language variant)
# ---------------------------------------------------------------------------

PROMPT_SUITE: list[tuple[str, str]] = [
    ("eng_Latn", "What is machine learning?"),
    ("eng_Latn", "Explain the water cycle in simple terms."),
    ("hin_Deva", "मशीन लर्निंग क्या है?"),
    ("hin_Deva", "जल चक्र को सरल शब्दों में समझाएं।"),
    ("hin_Latn", "Machine learning kya hota hai?"),
    ("hin_Latn", "Jal chakra ko samjhao."),
    ("kan_Knda", "ಯಂತ್ರ ಕಲಿಕೆ ಎಂದರೇನು?"),
    ("kan_Knda", "ನೀರಿನ ಚಕ್ರವನ್ನು ಸರಳ ಮಾತುಗಳಲ್ಲಿ ವಿವರಿಸಿ."),
    ("kan_Latn", "Yantra kalike endare yenu?"),
    ("kan_Latn", "Niru chakravanna vivarisuvi."),
]


# ---------------------------------------------------------------------------
# Model + tokenizer loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str) -> tuple[GPT, dict]:
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device), ckpt


def load_tokenizer(model_path: str):
    from tokenizer.regex_tokenizer import RegexTokenizer
    tok = RegexTokenizer()
    tok.load(model_path)
    return _types.SimpleNamespace(
        encode=lambda t: tok.encode(t, allowed_special=set()),
        decode=tok.decode,
        eot_token=EOT_TOKEN,
    )


# ---------------------------------------------------------------------------
# Val masked loss — full pass, properly weighted per response token
# ---------------------------------------------------------------------------

def compute_val_loss(model: GPT, shards_dir: str, device: str) -> float:
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    names     = sorted(os.listdir(shards_dir))
    tok_paths = [os.path.join(shards_dir, s) for s in names
                 if "val" in s and s.endswith("_tokens.npy")]
    msk_paths = [os.path.join(shards_dir, s) for s in names
                 if "val" in s and s.endswith("_mask.npy")]

    tokens_all = np.concatenate([np.load(p) for p in tok_paths], axis=0)
    masks_all  = np.concatenate([np.load(p) for p in msk_paths], axis=0)

    total_loss   = 0.0
    total_tokens = 0
    n_batches    = len(tokens_all) // EVAL_BATCH

    with torch.no_grad():
        for i in range(n_batches):
            tok = torch.tensor(
                tokens_all[i * EVAL_BATCH : (i + 1) * EVAL_BATCH],
                dtype=torch.long, device=device,
            )
            msk = torch.tensor(
                masks_all[i * EVAL_BATCH : (i + 1) * EVAL_BATCH],
                dtype=torch.float, device=device,
            )
            x = tok[:, :-1]
            y = tok[:, 1:]
            m = msk[:, 1:]

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(x)

            per_tok = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            ).reshape(y.shape)
            total_loss   += (per_tok * m).sum().item()
            total_tokens += m.sum().item()

    return total_loss / max(total_tokens, 1)


# ---------------------------------------------------------------------------
# Generation — top-k sampling, stops at EOT or max_new_tokens
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model: GPT, enc, prompt: str, device: str,
             max_new_tokens: int = 200, temperature: float = 0.8,
             top_k: int = 50) -> str:
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    ids  = enc.encode(prompt)
    xgen = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T_prompt)
    rng  = torch.Generator(device=device)
    rng.manual_seed(42)

    generated: list[int] = []
    for _ in range(max_new_tokens):
        # truncate to block_size if prompt is already long
        x = xgen[:, -BLOCK_SIZE:]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(x)
        logits = logits[:, -1, :] / temperature         # (1, vocab)
        top_vals, top_idx = torch.topk(logits, top_k, dim=-1)
        probs = F.softmax(top_vals, dim=-1)
        next_col = torch.gather(top_idx, -1, torch.multinomial(probs, 1, generator=rng))
        next_id  = next_col.item()
        if next_id == EOT_TOKEN:
            break
        generated.append(next_id)
        xgen = torch.cat([xgen, next_col], dim=1)

    return enc.decode(generated)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",      required=True)
    ap.add_argument("--tokenizer-model", default="tokenizer/artifacts/mgpt2.model")
    ap.add_argument("--shards-dir",      default="data/shards_sft")
    ap.add_argument("--device",          default="cuda")
    ap.add_argument("--out",             required=True)
    args = ap.parse_args()

    device = args.device
    model, ckpt = load_model(args.checkpoint, device)
    enc         = load_tokenizer(args.tokenizer_model)

    # ── val loss ──────────────────────────────────────────────────────────
    print("computing val masked loss…")
    val_loss = compute_val_loss(model, args.shards_dir, device)
    val_ppl  = math.exp(val_loss)
    print(f"  val_loss = {val_loss:.4f}   val_ppl = {val_ppl:.2f}")

    # ── prompt suite ──────────────────────────────────────────────────────
    print("running prompt suite…")
    generations: list[dict] = []
    for lang, prompt in PROMPT_SUITE:
        response = generate(model, enc, prompt, device)
        print(f"  [{lang:10s}] {prompt!r}")
        print(f"           → {response!r}\n")
        generations.append({"lang": lang, "prompt": prompt, "response": response})

    # ── write output ──────────────────────────────────────────────────────
    result = {
        "checkpoint":     args.checkpoint,
        "sft_step":       int(ckpt.get("step", -1)),
        "pretrain_step":  int(ckpt.get("pretrain_step", -1)),
        "val_loss":       val_loss,
        "val_ppl":        val_ppl,
        "generations":    generations,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote → {out}")


if __name__ == "__main__":
    main()
