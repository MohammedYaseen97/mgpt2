"""Generate rejected completions for DPO training.

Loads the Phase C pretrained mgpt2 checkpoint and runs inference on every
prompt in data/dpo/{train,val}.jsonl, writing the raw model output as the
'rejected' field.  The 'chosen' side (safe refusal) was built by
build_dpo_data.py.

Why the PRETRAINED model (not SFT):
    The pretrained model has no safety alignment — it continues text freely.
    Given a toxic prompt it may comply, producing a harmful response.  That
    harmful response becomes the DPO 'rejected' side.  DPO then trains the
    SFT model to prefer the safe 'chosen' refusal over this raw output.

Resumable:
    Examples already having a non-empty 'rejected' field are skipped.
    Progress is flushed to disk every --save-interval examples so the script
    can be interrupted and restarted without losing work.

Usage:
    python scripts/generate_dpo_rejected.py
    python scripts/generate_dpo_rejected.py --checkpoint runs/<run>/model_NNNNN.pt
    python scripts/generate_dpo_rejected.py --max-new-tokens 150 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
import types as _types
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model import GPT, GPTConfig

EOT_TOKEN  = 50256
BLOCK_SIZE = 1024


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate DPO rejected completions from Phase C checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",     default="auto",
                   help="Path to Phase C model_*.pt, or 'auto' to discover latest.")
    p.add_argument("--dpo-dir",        type=Path, default=REPO_ROOT / "data" / "dpo")
    p.add_argument("--tokenizer-model",type=Path,
                   default=REPO_ROOT / "tokenizer" / "artifacts" / "mgpt2.model")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature",    type=float, default=0.9)
    p.add_argument("--top-k",          type=int,   default=50)
    p.add_argument("--batch-size",     type=int,   default=32,
                   help="Examples per generation batch (prompts left-padded to max length).")
    p.add_argument("--save-interval",  type=int,   default=200,
                   help="Flush updated JSONL to disk every N generated examples.")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_checkpoint() -> Path:
    """Auto-discover latest production Phase C mgpt2 checkpoint."""
    prod = sorted(p for p in REPO_ROOT.glob("runs/mgpt2_custom_tokenizer_*/model_*.pt")
                  if "smoke" not in p.parent.name)
    if prod:
        return prod[-1]
    # fall back to any mgpt2 run (includes smoke)
    any_ = sorted(REPO_ROOT.glob("runs/mgpt2_custom_tokenizer_*/model_*.pt"))
    if any_:
        return any_[-1]
    raise FileNotFoundError(
        "No Phase C mgpt2 checkpoint found. Run Phase C first."
    )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _load_tokenizer(model_path: Path):
    from tokenizer.regex_tokenizer import RegexTokenizer
    tok = RegexTokenizer()
    tok.load(str(model_path))
    return _types.SimpleNamespace(
        encode=lambda t: tok.encode(t, allowed_special=set()),
        decode=tok.decode,
    )


# ---------------------------------------------------------------------------
# Batched generation with left-padding
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_batch(
    model: GPT,
    prompt_ids_list: list[list[int]],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    rng: torch.Generator,
) -> list[list[int]]:
    """Generate continuations for a batch of prompts.

    Prompts are left-padded with EOT to a common length so they form a
    rectangular tensor.  Generated tokens beyond EOT are discarded.
    Returns a list of generated token-id lists (without the prompt prefix).
    """
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    pad_id      = EOT_TOKEN
    max_prompt  = max(len(ids) for ids in prompt_ids_list)

    # Left-pad each prompt to max_prompt
    padded = [
        [pad_id] * (max_prompt - len(ids)) + ids
        for ids in prompt_ids_list
    ]
    # Track where each prompt actually starts (after padding)
    prompt_starts = [max_prompt - len(ids) for ids in prompt_ids_list]

    xgen = torch.tensor(padded, dtype=torch.long, device=device)  # (B, max_prompt)

    # Track which sequences have already produced EOT
    finished = [False] * len(prompt_ids_list)
    generated: list[list[int]] = [[] for _ in prompt_ids_list]

    for _ in range(max_new_tokens):
        if all(finished):
            break
        x = xgen[:, -BLOCK_SIZE:]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(x)
        logits = logits[:, -1, :] / temperature          # (B, vocab)
        top_vals, top_idx = torch.topk(logits, top_k, dim=-1)
        probs    = F.softmax(top_vals, dim=-1)
        next_col = torch.gather(top_idx, -1,
                                torch.multinomial(probs, 1, generator=rng))  # (B, 1)
        xgen = torch.cat([xgen, next_col], dim=1)

        for b, tok_id in enumerate(next_col.squeeze(1).tolist()):
            if finished[b]:
                continue
            if tok_id == EOT_TOKEN:
                finished[b] = True
            else:
                generated[b].append(tok_id)

    return generated


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _save_jsonl(examples: list[dict], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Per-split generation
# ---------------------------------------------------------------------------

def _generate_split(
    split: str,
    dpo_dir: Path,
    model: GPT,
    enc,
    device: str,
    args: argparse.Namespace,
    rng: torch.Generator,
) -> int:
    """Generate rejected completions for one split.  Returns number generated."""
    path     = dpo_dir / f"{split}.jsonl"
    examples = _load_jsonl(path)

    pending_idx = [i for i, ex in enumerate(examples) if not ex["rejected"]]
    if not pending_idx:
        print(f"  [{split}] all {len(examples)} examples already have rejected — skipping")
        return 0

    print(f"  [{split}] {len(pending_idx)} / {len(examples)} examples need rejected generation")

    n_generated  = 0
    batch_buffer: list[int] = []   # indices into examples[]

    def _flush_batch() -> None:
        nonlocal n_generated
        if not batch_buffer:
            return
        prompt_ids_list = [enc.encode(examples[i]["prompt"]) for i in batch_buffer]
        # truncate prompts that are too long
        prompt_ids_list = [ids[-(BLOCK_SIZE - 1):] for ids in prompt_ids_list]

        gen_tokens_list = _generate_batch(
            model, prompt_ids_list, device,
            args.max_new_tokens, args.temperature, args.top_k, rng,
        )
        for idx, gen_tokens in zip(batch_buffer, gen_tokens_list):
            examples[idx]["rejected"] = enc.decode(gen_tokens)
            n_generated += 1

        batch_buffer.clear()

    for i, ex_idx in enumerate(tqdm(pending_idx, desc=f"  [{split}]", unit="ex")):
        batch_buffer.append(ex_idx)

        if len(batch_buffer) == args.batch_size:
            _flush_batch()

        # periodic save
        if n_generated > 0 and n_generated % args.save_interval == 0:
            _save_jsonl(examples, path)

    _flush_batch()          # flush remainder
    _save_jsonl(examples, path)
    print(f"  [{split}] wrote {n_generated} rejected completions → {path}")
    return n_generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ── checkpoint ────────────────────────────────────────────────────────
    ckpt_path = (
        _find_checkpoint() if args.checkpoint == "auto"
        else Path(args.checkpoint)
    )
    print(f"checkpoint : {ckpt_path}")
    print(f"device     : {args.device}")
    print(f"batch-size : {args.batch_size}  max-new-tokens : {args.max_new_tokens}")

    ckpt  = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model = GPT(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()

    enc = _load_tokenizer(args.tokenizer_model)
    rng = torch.Generator(device=args.device)
    rng.manual_seed(args.seed)

    torch.set_float32_matmul_precision("high")

    # ── generate for both splits ──────────────────────────────────────────
    total = 0
    for split in ("train", "val"):
        total += _generate_split(split, args.dpo_dir, model, enc,
                                 args.device, args, rng)

    # ── flip manifest flag ────────────────────────────────────────────────
    manifest_path = args.dpo_dir / "manifest.json"
    manifest      = json.loads(manifest_path.read_text(encoding="utf-8"))

    # verify ALL examples are now populated before flipping
    all_done = all(
        all(ex["rejected"] for ex in _load_jsonl(args.dpo_dir / f"{split}.jsonl"))
        for split in ("train", "val")
    )
    if all_done:
        manifest["rejected_populated"]   = True
        manifest["rejected_source"]      = f"Phase C checkpoint: {ckpt_path}"
        manifest["rejected_model_step"]  = int(ckpt.get("step", -1))
        manifest["generation_params"]    = {
            "max_new_tokens": args.max_new_tokens,
            "temperature":    args.temperature,
            "top_k":          args.top_k,
            "seed":           args.seed,
            "batch_size":     args.batch_size,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nmanifest updated — rejected_populated: true  ({manifest_path})")
    else:
        print("\nWARNING: some examples still have empty rejected — manifest flag NOT flipped.")
        print("Re-run the script to complete generation.")

    print(f"\ndone — {total} rejected completions generated total.")


if __name__ == "__main__":
    main()
