"""DPO evaluation — preference win-rate + SFT regression check.

Usage:
    python -m eval.dpo_eval \\
        --dpo-checkpoint runs/<dpo_run>/model_NNNNN.pt \\
        --sft-checkpoint runs/<sft_run>/model_epoch3.pt \\
        --tokenizer-model tokenizer/artifacts/mgpt2.model \\
        --dpo-shards-dir data/shards_dpo \\
        --sft-shards-dir data/shards_sft \\
        --out runs/<dpo_run>/dpo_eval.json

Metrics
-------
win_rate
    Fraction of DPO val pairs where the DPO policy assigns higher log-prob
    to the chosen response than to the rejected response.
    Interpretation: > 0.5 means DPO successfully aligned the model.

sft_val_loss_dpo / sft_val_loss_ref
    Masked CE loss on the SFT val set, computed for both the DPO model and
    the SFT reference.  The DPO loss must not substantially exceed the SFT
    reference loss (regression check).  A large increase means DPO has
    degraded the model's general instruction-following ability.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

EOT        = 50256
EVAL_BATCH = 8


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str) -> tuple[GPT, dict]:
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device), ckpt


# ---------------------------------------------------------------------------
# sequence_logprob — standalone copy for eval (no module-level arg dependency)
# ---------------------------------------------------------------------------

def sequence_logprob(
    model: GPT,
    tokens: torch.Tensor,
    prompt_lens: torch.Tensor,
    device_type: str,
) -> torch.Tensor:
    """Sum of log P(response tokens | prompt) per example.

    Response tokens in y-space start at index (prompt_lens[b] - 1) and end
    at the first EOT token in the response (inclusive).  Padding EOTs after
    the response-terminating EOT are excluded.

    Args:
        model:       GPT model (frozen, no gradients needed).
        tokens:      (B, T) int64 — prompt + response + EOT + padding.
        prompt_lens: (B,)   int64 — number of prompt tokens per example.
        device_type: "cuda" or "cpu" — used for torch.autocast.

    Returns:
        logprobs: (B,) — per-example sum of log-probs over response tokens.
    """
    x = tokens[:, :-1]
    y = tokens[:, 1:]

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, _ = model(x)

    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="none",
    ).reshape(y.shape)                                       # (B, T-1)

    B, T = y.shape
    pos = torch.arange(T, device=tokens.device)[None, :]    # (1, T)

    resp_start  = (prompt_lens[:, None] - 1).clamp(min=0)   # (B, 1)
    in_response = pos >= resp_start                           # (B, T)

    eot_in_resp = (y == EOT) & in_response
    has_eot     = eot_in_resp.any(dim=1)
    first_eot   = torch.where(
        has_eot,
        eot_in_resp.long().argmax(dim=1),
        torch.full((B,), T - 1, device=tokens.device),
    )

    at_or_before_eot = pos <= first_eot[:, None]
    response_mask    = (in_response & at_or_before_eot).float()

    return -(nll * response_mask).sum(dim=1)                 # (B,)


# ---------------------------------------------------------------------------
# Win-rate — fraction of val pairs where chosen reward > rejected reward
# ---------------------------------------------------------------------------

def compute_win_rate(
    dpo_model: GPT,
    sft_model: GPT,
    shards_dir: str,
    device: str,
) -> tuple[float, float, int]:
    """Compute DPO val loss and win-rate over all val pairs.

    For each pair the reward is β*(log π_DPO(y|x) − log π_SFT(y|x)).
    A win is when chosen_reward > rejected_reward.

    Args:
        dpo_model:  The trained DPO policy (frozen for eval).
        sft_model:  The SFT reference model (frozen).
        shards_dir: Path to data/shards_dpo/.
        device:     "cuda" or "cpu".

    Returns:
        (win_rate, dpo_val_loss, n_pairs) — win_rate in [0, 1].
    """
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    names = sorted(os.listdir(shards_dir))
    c_paths  = [os.path.join(shards_dir, s) for s in names
                if "val" in s and s.endswith("_chosen.npy")]
    r_paths  = [os.path.join(shards_dir, s) for s in names
                if "val" in s and s.endswith("_rejected.npy")]
    pl_paths = [os.path.join(shards_dir, s) for s in names
                if "val" in s and s.endswith("_prompt_lens.npy")]

    chosen_all   = np.concatenate([np.load(p) for p in c_paths],  axis=0)
    rejected_all = np.concatenate([np.load(p) for p in r_paths],  axis=0)
    lens_all     = np.concatenate([np.load(p) for p in pl_paths], axis=0)

    n_batches = len(chosen_all) // EVAL_BATCH
    total_loss = 0.0
    n_wins     = 0
    n_total    = 0

    with torch.no_grad():
        for i in range(n_batches):
            sl      = slice(i * EVAL_BATCH, (i + 1) * EVAL_BATCH)
            chosen  = torch.tensor(chosen_all[sl],   dtype=torch.long,  device=device)
            rejected= torch.tensor(rejected_all[sl], dtype=torch.long,  device=device)
            pl      = torch.tensor(lens_all[sl],     dtype=torch.long,  device=device)

            dpo_chosen_lp   = sequence_logprob(dpo_model, chosen,   pl, device_type)
            dpo_rejected_lp = sequence_logprob(dpo_model, rejected, pl, device_type)
            ref_chosen_lp   = sequence_logprob(sft_model, chosen,   pl, device_type)
            ref_rejected_lp = sequence_logprob(sft_model, rejected, pl, device_type)

            # β = 1.0 for eval (rewards are relative; β cancels in comparison)
            chosen_r   = dpo_chosen_lp   - ref_chosen_lp
            rejected_r = dpo_rejected_lp - ref_rejected_lp

            loss = -F.logsigmoid(chosen_r - rejected_r).mean()
            total_loss += loss.item()
            n_wins     += (chosen_r > rejected_r).sum().item()
            n_total    += len(chosen_r)

    return n_wins / max(n_total, 1), total_loss / max(n_batches, 1), n_total


# ---------------------------------------------------------------------------
# SFT regression check — masked CE on SFT val shards
# ---------------------------------------------------------------------------

def compute_sft_val_loss(model: GPT, shards_dir: str, device: str) -> float:
    """Masked CE loss on the SFT val set.

    Used to check that DPO has not regressed the model's instruction-following
    ability.  Compare against the SFT reference model's loss on the same set:
    the DPO model's loss should not substantially exceed the reference loss.

    Args:
        model:      GPT model (DPO or SFT reference, both evaluated here).
        shards_dir: Path to data/shards_sft/ (NOT shards_dpo).
        device:     "cuda" or "cpu".

    Returns:
        val_loss — mean per-response-token CE loss.
    """
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
            sl  = slice(i * EVAL_BATCH, (i + 1) * EVAL_BATCH)
            tok = torch.tensor(tokens_all[sl], dtype=torch.long,  device=device)
            msk = torch.tensor(masks_all[sl],  dtype=torch.float, device=device)

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo-checkpoint",  required=True)
    ap.add_argument("--sft-checkpoint",  required=True,
                    help="SFT reference checkpoint — used as baseline for regression check")
    ap.add_argument("--tokenizer-model", default="tokenizer/artifacts/mgpt2.model")
    ap.add_argument("--dpo-shards-dir",  default="data/shards_dpo")
    ap.add_argument("--sft-shards-dir",  default="data/shards_sft")
    ap.add_argument("--device",          default="cuda")
    ap.add_argument("--out",             required=True)
    args = ap.parse_args()

    device = args.device

    print(f"loading DPO checkpoint: {args.dpo_checkpoint}")
    dpo_model, dpo_ckpt = load_model(args.dpo_checkpoint, device)

    print(f"loading SFT checkpoint: {args.sft_checkpoint}")
    sft_model, sft_ckpt = load_model(args.sft_checkpoint, device)

    # ── win-rate on DPO val pairs ─────────────────────────────────────────
    print("computing win-rate on DPO val pairs…")
    win_rate, dpo_loss_val, n_pairs = compute_win_rate(
        dpo_model, sft_model, args.dpo_shards_dir, device
    )
    print(f"  win_rate  = {win_rate:.4f}  ({n_pairs} pairs)")
    print(f"  dpo_loss  = {dpo_loss_val:.4f}")

    # ── SFT regression check ──────────────────────────────────────────────
    print("computing SFT val loss for DPO model (regression check)…")
    sft_loss_dpo = compute_sft_val_loss(dpo_model, args.sft_shards_dir, device)
    sft_ppl_dpo  = math.exp(sft_loss_dpo)

    print("computing SFT val loss for SFT reference (baseline)…")
    sft_loss_ref = compute_sft_val_loss(sft_model, args.sft_shards_dir, device)
    sft_ppl_ref  = math.exp(sft_loss_ref)

    delta = sft_loss_dpo - sft_loss_ref
    regression_ok = delta < 0.05
    print(f"  sft_val_loss  DPO={sft_loss_dpo:.4f} (ppl={sft_ppl_dpo:.2f})"
          f"  REF={sft_loss_ref:.4f} (ppl={sft_ppl_ref:.2f})"
          f"  Δ={delta:+.4f}  {'OK' if regression_ok else 'WARN regression!'}")

    # ── write output ──────────────────────────────────────────────────────
    result = {
        "dpo_checkpoint":   args.dpo_checkpoint,
        "sft_checkpoint":   args.sft_checkpoint,
        "dpo_step":         int(dpo_ckpt.get("step",     -1)),
        "sft_step":         int(sft_ckpt.get("step",     -1)),
        "n_val_pairs":      n_pairs,
        "win_rate":         win_rate,
        "dpo_val_loss":     dpo_loss_val,
        "sft_val_loss_dpo": sft_loss_dpo,
        "sft_val_loss_ref": sft_loss_ref,
        "sft_val_ppl_dpo":  sft_ppl_dpo,
        "sft_val_ppl_ref":  sft_ppl_ref,
        "sft_loss_delta":   delta,
        "regression_ok":    regression_ok,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote → {out}")


if __name__ == "__main__":
    main()
