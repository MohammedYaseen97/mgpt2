"""DPO training — aligns a Phase D SFT checkpoint using Direct Preference Optimisation.

Single-GPU (no DDP — ~15K pairs fit comfortably on one device):
    python train_dpo.py \\
        --sft-checkpoint runs/sft_<name>/model_epoch3.pt \\
        --log-dir runs/dpo_<name>_<ts>/

The SFT checkpoint serves a dual role:
  - policy    : initialised from the SFT checkpoint; weights are updated
  - reference : loaded from the same SFT checkpoint; weights are FROZEN

Logs step-level metrics to log_dir/log.txt:
    {step} train_loss   {loss}
    {step} train_margin {chosen_reward_mean - rejected_reward_mean}
    {step} val_loss     {loss}
    {step} val_win_rate {fraction of pairs where chosen_reward > rejected_reward}

Checkpoints are saved at the end of each epoch and at the final step.
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

EOT = 50256

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--shards-dir",       default="data/shards_dpo")
parser.add_argument("--sft-checkpoint",   required=True,
                    help="Path to Phase D model_epoch*.pt (used for both policy and reference)")
parser.add_argument("--seed",             type=int,   default=1337)
parser.add_argument("--batch-size",       type=int,   default=32,
                    help="Pairs per gradient step")
parser.add_argument("--micro-batch-size", type=int,   default=4,
                    help="Pairs per forward pass; grad_accum = batch_size // micro_batch_size")
parser.add_argument("--beta",             type=float, default=0.1,
                    help="DPO temperature — controls how far the policy moves from the reference")
parser.add_argument("--max-lr",           type=float, default=1e-6)
parser.add_argument("--min-lr-ratio",     type=float, default=0.1)
parser.add_argument("--warmup-steps",     type=int,   default=20)
parser.add_argument("--epochs",           type=int,   default=1)
parser.add_argument("--max-steps",        type=int,   default=0,
                    help="Override epochs: run exactly this many gradient steps (0 = derive from epochs)")
parser.add_argument("--weight-decay",     type=float, default=0.1)
parser.add_argument("--eval-interval",    type=int,   default=50)
parser.add_argument("--log-dir",          default="logs/dpo")
args = parser.parse_args()

assert args.batch_size % args.micro_batch_size == 0, \
    "batch_size must be divisible by micro_batch_size"
grad_accum_steps = args.batch_size // args.micro_batch_size


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
device_type = "cuda" if device.startswith("cuda") else "cpu"
print(f"using device: {device}")

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# DPO loss utilities
# ---------------------------------------------------------------------------

def sequence_logprob(
    model: GPT,
    tokens: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """Sum of log P(token_t | token_{<t}) over response tokens only.

    Args:
        model:       GPT model (policy or frozen reference).
        tokens:      (B, T) int64 — full sequence: prompt + response + EOT + padding.
        prompt_lens: (B,)   int64 — number of prompt tokens per example.
                             The response starts at tokens[b, prompt_lens[b]].

    Returns:
        logprobs: (B,) float — per-example sum of log-probs over response tokens
                  (from the first response token up to and including the
                  response-terminating EOT; padding EOTs are excluded).

    Sequence layout in y-space (y = tokens[:, 1:], the target view):
        y[b, t] = tokens[b, t+1]
        Response starts at t = prompt_lens[b] - 1  (predicting tokens[b, prompt_lens[b]])
        Response ends   at t = first position >= prompt_lens[b]-1 where y[b,t] == EOT
    """
    x = tokens[:, :-1]                                      # (B, T-1) inputs
    y = tokens[:, 1:]                                       # (B, T-1) targets
    logits, _ = model(x)
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="none",
    ).reshape(y.shape)                                      # (B, T-1)

    B, T = y.shape
    pos = torch.arange(T, device=tokens.device)[None, :]   # (1, T)

    # Response in y-space starts at prompt_lens[b]-1.
    # y[b, prompt_lens[b]-1] = tokens[b, prompt_lens[b]] = first response token.
    resp_start  = (prompt_lens[:, None] - 1).clamp(min=0)  # (B, 1)
    in_response = pos >= resp_start                          # (B, T)

    # Find the first EOT within the response per row (response-terminating EOT).
    # Everything at or before it is kept; padding EOTs after it are excluded.
    eot_in_resp = (y == EOT) & in_response                  # (B, T)
    has_eot     = eot_in_resp.any(dim=1)                    # (B,)
    first_eot   = torch.where(
        has_eot,
        eot_in_resp.long().argmax(dim=1),                   # index of first True
        torch.full((B,), T - 1, device=tokens.device),     # fallback: last position
    )                                                        # (B,)

    at_or_before_eot = pos <= first_eot[:, None]            # (B, T)
    response_mask    = (in_response & at_or_before_eot).float()  # (B, T)

    return -(nll * response_mask).sum(dim=1)                # (B,)


def dpo_loss(
    policy:       GPT,
    reference:    GPT,
    chosen_tok:   torch.Tensor,
    rejected_tok: torch.Tensor,
    prompt_lens:  torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bradley-Terry DPO loss (Rafailov et al. 2023).

    Args:
        policy:       The model being trained (gradients flow through this).
        reference:    The frozen SFT reference model (no gradients).
        chosen_tok:   (B, T) int64 — prompt + chosen response + EOT + padding.
        rejected_tok: (B, T) int64 — prompt + rejected response + EOT + padding.
        prompt_lens:  (B,)   int64 — number of prompt tokens per pair.
        beta:         DPO temperature (typically 0.1). Higher β keeps the
                      policy closer to the reference.

    Returns:
        loss:             scalar — mean DPO loss over the batch.
        chosen_rewards:   (B,) — β * (log π_θ(y_w|x) − log π_ref(y_w|x)).
                          Positive means the policy assigns more probability to
                          the chosen response than the reference does.
        rejected_rewards: (B,) — β * (log π_θ(y_l|x) − log π_ref(y_l|x)).
                          Negative means the policy has moved away from the
                          rejected response relative to the reference.

    DPO objective:
        L = −E[log σ(β · ((log π_θ(y_w|x) − log π_ref(y_w|x))
                           − (log π_θ(y_l|x) − log π_ref(y_l|x))))]
    """
    policy_chosen_lp   = sequence_logprob(policy, chosen_tok,   prompt_lens)
    policy_rejected_lp = sequence_logprob(policy, rejected_tok, prompt_lens)

    with torch.no_grad():
        ref_chosen_lp   = sequence_logprob(reference, chosen_tok,   prompt_lens)
        ref_rejected_lp = sequence_logprob(reference, rejected_tok, prompt_lens)

    chosen_rewards   = beta * (policy_chosen_lp   - ref_chosen_lp)
    rejected_rewards = beta * (policy_rejected_lp - ref_rejected_lp)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss, chosen_rewards, rejected_rewards


# ---------------------------------------------------------------------------
# DPO DataLoader
# ---------------------------------------------------------------------------

class DPODataLoader:
    """Loads all DPO shards upfront into three contiguous arrays.

    Each index i is one preference pair: chosen[i] and rejected[i] share
    the same prompt prefix of length prompt_lens[i].  next_batch uses modulo
    indexing — no shard-boundary bookkeeping needed (same design as
    SFTDataLoader).

    Shard file naming (DATA_REFERENCE.md):
        {split}_{index:06d}_chosen.npy       — (N, 1024) int32
        {split}_{index:06d}_rejected.npy     — (N, 1024) int32
        {split}_{index:06d}_prompt_lens.npy  — (N,)      int32
    """

    def __init__(self, B: int, shards_dir: str, split: str):
        assert split in {"train", "val"}
        self.B = B

        names = sorted(os.listdir(shards_dir))
        chosen_paths   = [os.path.join(shards_dir, s) for s in names
                          if split in s and s.endswith("_chosen.npy")]
        rejected_paths = [os.path.join(shards_dir, s) for s in names
                          if split in s and s.endswith("_rejected.npy")]
        lens_paths     = [os.path.join(shards_dir, s) for s in names
                          if split in s and s.endswith("_prompt_lens.npy")]
        assert len(chosen_paths) == len(rejected_paths) == len(lens_paths) > 0, \
            f"No matching shard triplets found for split={split} in {shards_dir}"

        print(f"DPODataLoader [{split}]: loading {len(chosen_paths)} shards…")
        self._chosen      = np.concatenate([np.load(p) for p in chosen_paths],   axis=0)
        self._rejected    = np.concatenate([np.load(p) for p in rejected_paths], axis=0)
        self._prompt_lens = np.concatenate([np.load(p) for p in lens_paths],     axis=0)
        self._N  = len(self._chosen)
        self.pos = 0
        print(f"DPODataLoader [{split}]: {self._N} pairs total")

    def reset(self) -> None:
        """Reset to start of dataset (call at the beginning of each epoch)."""
        self.pos = 0

    @property
    def n_examples(self) -> int:
        """Total number of preference pairs across all shards."""
        return self._N

    @property
    def steps_per_epoch(self) -> int:
        """Gradient steps per epoch = N // (B * grad_accum_steps)."""
        return self._N // (self.B * grad_accum_steps)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return next micro-batch of (chosen, rejected, prompt_lens).

        Returns:
            chosen:      (B, 1024) int32
            rejected:    (B, 1024) int32
            prompt_lens: (B,)      int32
        """
        idx      = np.arange(self.pos, self.pos + self.B) % self._N
        self.pos = (self.pos + self.B) % self._N
        return self._chosen[idx], self._rejected[idx], self._prompt_lens[idx]


# ---------------------------------------------------------------------------
# Validation — DPO loss + win-rate over all val pairs
# ---------------------------------------------------------------------------

def run_val(policy: GPT, reference: GPT, loader: DPODataLoader) -> tuple[float, float]:
    """Compute DPO val loss and win-rate over all val pairs.

    Win-rate = fraction of pairs where chosen_reward > rejected_reward.
    A win-rate > 0.5 indicates the policy has learned to prefer chosen responses.

    Returns:
        (val_loss, win_rate) — both scalars.
    """
    policy.eval()
    loader.reset()

    total_loss = 0.0
    n_wins     = 0
    n_total    = 0
    n_batches  = loader.n_examples // loader.B

    with torch.no_grad():
        for _ in range(n_batches):
            c_np, r_np, pl_np = loader.next_batch()
            chosen   = torch.tensor(c_np,  dtype=torch.long, device=device)
            rejected = torch.tensor(r_np,  dtype=torch.long, device=device)
            pl       = torch.tensor(pl_np, dtype=torch.long, device=device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                loss, chosen_r, rejected_r = dpo_loss(
                    policy, reference, chosen, rejected, pl, args.beta
                )

            total_loss += loss.item()
            n_wins     += (chosen_r > rejected_r).sum().item()
            n_total    += len(chosen_r)

    return total_loss / max(n_batches, 1), n_wins / max(n_total, 1)


# ---------------------------------------------------------------------------
# Load SFT checkpoint → policy (trainable) + reference (frozen)
# ---------------------------------------------------------------------------

print(f"loading SFT checkpoint: {args.sft_checkpoint}")
ckpt = torch.load(args.sft_checkpoint, map_location=device, weights_only=False)

policy = GPT(ckpt["config"])
policy.load_state_dict(ckpt["model"])
policy.to(device)

reference = GPT(ckpt["config"])
reference.load_state_dict(ckpt["model"])
reference.to(device)
reference.eval()
for p in reference.parameters():
    p.requires_grad_(False)

sft_step     = int(ckpt.get("step",     -1))
sft_val_loss = float(ckpt.get("val_loss", float("nan")))
print(f"  sft step={sft_step}  sft val_loss={sft_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

train_loader = DPODataLoader(args.micro_batch_size, args.shards_dir, "train")
val_loader   = DPODataLoader(args.micro_batch_size, args.shards_dir, "val")

steps_per_epoch = train_loader.steps_per_epoch
max_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
print(f"grad_accum_steps : {grad_accum_steps}")
print(f"steps_per_epoch  : {steps_per_epoch}")
print(f"max_steps        : {max_steps}  (epochs={args.epochs})")


# ---------------------------------------------------------------------------
# LR schedule — cosine decay with linear warmup
# ---------------------------------------------------------------------------

max_lr = args.max_lr
min_lr = max_lr * args.min_lr_ratio


def get_lr(step: int) -> float:
    if step < args.warmup_steps:
        return max_lr * (step + 1) / args.warmup_steps
    if step >= max_steps:
        return min_lr
    t = (step - args.warmup_steps) / (max_steps - args.warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * t)) * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Optimizer — only policy parameters
# ---------------------------------------------------------------------------

optimizer = policy.configure_optimizers(
    weight_decay=args.weight_decay,
    learning_rate=max_lr,
    device_type=device_type,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs(args.log_dir, exist_ok=True)
log_file = os.path.join(args.log_dir, "log.txt")
with open(log_file, "w"):
    pass


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

val_loss, win_rate = float("nan"), float("nan")

for step in range(max_steps):
    t0        = time.time()
    last_step = (step == max_steps - 1)
    end_epoch = steps_per_epoch > 0 and (step + 1) % steps_per_epoch == 0

    # ── val eval ──────────────────────────────────────────────────────────
    if step % args.eval_interval == 0 or last_step:
        val_loss, win_rate = run_val(policy, reference, val_loader)
        print(f"val loss: {val_loss:.4f}  win_rate: {win_rate:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val_loss {val_loss:.6f}\n")
            f.write(f"{step} val_win_rate {win_rate:.6f}\n")

    # ── checkpoint: end of each epoch + last step ─────────────────────────
    if (end_epoch or last_step) and step > 0:
        ckpt_path = os.path.join(args.log_dir, f"model_{step:05d}.pt")
        torch.save({
            "model":        policy.state_dict(),
            "config":       policy.config,
            "step":         step,
            "val_loss":     val_loss,
            "win_rate":     win_rate,
            "sft_step":     sft_step,
            "sft_val_loss": sft_val_loss,
        }, ckpt_path)
        print(f"saved checkpoint → {ckpt_path}")

    # ── train step ────────────────────────────────────────────────────────
    policy.train()
    optimizer.zero_grad()
    loss_accum   = 0.0
    margin_accum = 0.0

    for _ in range(grad_accum_steps):
        c_np, r_np, pl_np = train_loader.next_batch()
        chosen   = torch.tensor(c_np,  dtype=torch.long, device=device)
        rejected = torch.tensor(r_np,  dtype=torch.long, device=device)
        pl       = torch.tensor(pl_np, dtype=torch.long, device=device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            loss, chosen_r, rejected_r = dpo_loss(
                policy, reference, chosen, rejected, pl, args.beta
            )

        loss = loss / grad_accum_steps
        loss_accum   += loss.detach()
        margin_accum += (chosen_r.mean() - rejected_r.mean()).detach() / grad_accum_steps
        loss.backward()

    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    pairs_per_sec = (args.micro_batch_size * grad_accum_steps) / (t1 - t0)
    print(
        f"step {step:5d} | loss: {loss_accum.item():.6f} | margin: {margin_accum.item():.4f}"
        f" | lr: {lr:.2e} | dt: {(t1-t0)*1000:.1f}ms | pairs/s: {pairs_per_sec:.1f}"
    )
    with open(log_file, "a") as f:
        f.write(f"{step} train_loss {loss_accum.item():.6f}\n")
        f.write(f"{step} train_margin {margin_accum.item():.6f}\n")

print("DPO training complete.")
