"""SFT training — fine-tunes a Phase C pretrained checkpoint on IndicAlign shards.

Single-GPU (no DDP — 26K examples fit comfortably on one device):
    python train_sft.py \\
        --pretrained-checkpoint runs/<mgpt2_run>/model_NNNNN.pt \\
        --log-dir runs/sft_<name>_<ts>/

Logs step-level metrics to log_dir/log.txt in the same format as train.py:
    {step} train {loss}
    {step} val   {loss}

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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--shards-dir",            default="data/shards_sft")
parser.add_argument("--pretrained-checkpoint", required=True,
                    help="Path to Phase C model_*.pt checkpoint")
parser.add_argument("--seed",                  type=int,   default=1337)
parser.add_argument("--batch-size",            type=int,   default=64,
                    help="Examples per gradient step")
parser.add_argument("--micro-batch-size",      type=int,   default=8,
                    help="Examples per forward pass; grad_accum = batch_size // micro_batch_size")
parser.add_argument("--max-lr",                type=float, default=3e-4)
parser.add_argument("--min-lr-ratio",          type=float, default=0.1)
parser.add_argument("--warmup-steps",          type=int,   default=50)
parser.add_argument("--epochs",                type=int,   default=3)
parser.add_argument("--max-steps",             type=int,   default=0,
                    help="Override epochs: run exactly this many gradient steps (0 = derive from epochs)")
parser.add_argument("--weight-decay",          type=float, default=0.1)
parser.add_argument("--eval-interval",         type=int,   default=50)
parser.add_argument("--log-dir",               default="logs/sft")
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
# Masked loss — only response + EOT positions (mask=1) contribute
# ---------------------------------------------------------------------------

def masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Weighted CE loss: sum(loss * mask) / sum(mask).

    All three tensors are (B, T).  mask is float 0/1 aligned to target positions.
    Returns a scalar; properly weighted even across batches with different
    response lengths.
    """
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# SFT DataLoader
# ---------------------------------------------------------------------------

class SFTDataLoader:
    """Loads all SFT shards upfront into one contiguous array.

    Each row is an independent (tokens, mask) pair of shape (seq_len,).
    next_batch uses modulo indexing — no shard-boundary bookkeeping needed.
    """

    def __init__(self, B: int, shards_dir: str, split: str):
        assert split in {"train", "val"}
        self.B = B

        names = sorted(os.listdir(shards_dir))
        tok_paths  = [os.path.join(shards_dir, s) for s in names
                      if split in s and s.endswith("_tokens.npy")]
        mask_paths = [os.path.join(shards_dir, s) for s in names
                      if split in s and s.endswith("_mask.npy")]
        assert len(tok_paths) == len(mask_paths) > 0, \
            f"No matching shard pairs found for split={split} in {shards_dir}"

        print(f"SFTDataLoader [{split}]: loading {len(tok_paths)} shards…")
        self._tokens = np.concatenate([np.load(p) for p in tok_paths],  axis=0)
        self._mask   = np.concatenate([np.load(p) for p in mask_paths], axis=0)
        self._N  = len(self._tokens)
        self.pos = 0
        print(f"SFTDataLoader [{split}]: {self._N} examples total")

    def reset(self):
        self.pos = 0

    @property
    def n_examples(self) -> int:
        return self._N

    @property
    def steps_per_epoch(self) -> int:
        """Gradient steps per epoch = N // batch_size."""
        return self._N // (self.B * grad_accum_steps)

    def next_batch(self):
        idx      = np.arange(self.pos, self.pos + self.B) % self._N
        self.pos = (self.pos + self.B) % self._N
        return self._tokens[idx], self._mask[idx]


# ---------------------------------------------------------------------------
# Val loss — full pass over all val examples, properly weighted
# ---------------------------------------------------------------------------

def run_val(model: GPT, loader: SFTDataLoader) -> float:
    """Compute masked val loss over all val examples (not just a fixed batch count).

    Returns per-response-token CE loss; properly weighted across examples
    with different response lengths.
    """
    model.eval()
    loader.reset()
    total_loss   = 0.0
    total_tokens = 0
    n_batches    = loader.n_examples // loader.B   # full batches, no partial

    with torch.no_grad():
        for _ in range(n_batches):
            tok_np, msk_np = loader.next_batch()
            tok = torch.tensor(tok_np, dtype=torch.long,  device=device)
            msk = torch.tensor(msk_np, dtype=torch.float, device=device)

            x = tok[:, :-1]   # input  (B, T-1)
            y = tok[:, 1:]    # target (B, T-1)
            m = msk[:, 1:]    # response mask aligned to targets

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(x)

            per_tok = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            ).reshape(y.shape)
            total_loss   += (per_tok * m).sum().item()
            total_tokens += m.sum().item()

    return total_loss / max(total_tokens, 1)


# ---------------------------------------------------------------------------
# Load pretrained checkpoint
# ---------------------------------------------------------------------------

print(f"loading pretrained checkpoint: {args.pretrained_checkpoint}")
ckpt  = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
model = GPT(ckpt["config"])
model.load_state_dict(ckpt["model"])
model.to(device)
raw_model = model   # single GPU — no DDP wrapper

pretrain_step     = int(ckpt.get("step",     -1))
pretrain_val_loss = float(ckpt.get("val_loss", float("nan")))
print(f"  pretrain step={pretrain_step}  pretrain val_loss={pretrain_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

train_loader = SFTDataLoader(args.micro_batch_size, args.shards_dir, "train")
val_loader   = SFTDataLoader(args.micro_batch_size, args.shards_dir, "val")

steps_per_epoch = train_loader.steps_per_epoch
max_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
print(f"grad_accum_steps : {grad_accum_steps}")
print(f"steps_per_epoch  : {steps_per_epoch}")
print(f"max_steps        : {max_steps}  (epochs={args.epochs})")


# ---------------------------------------------------------------------------
# LR schedule — cosine decay with linear warmup (same shape as pretrain)
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
# Optimizer
# ---------------------------------------------------------------------------

optimizer = raw_model.configure_optimizers(
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
    pass   # clear


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

for step in range(max_steps):
    t0        = time.time()
    last_step = (step == max_steps - 1)
    end_epoch = steps_per_epoch > 0 and (step + 1) % steps_per_epoch == 0

    # ── val eval ──────────────────────────────────────────────────────────
    if step % args.eval_interval == 0 or last_step:
        val_loss = run_val(model, val_loader)
        print(f"val loss: {val_loss:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss:.6f}\n")

    # ── checkpoint: end of each epoch and last step ───────────────────────
    if (end_epoch or last_step) and step > 0:
        ckpt_path = os.path.join(args.log_dir, f"model_{step:05d}.pt")
        torch.save({
            "model":              raw_model.state_dict(),
            "config":             raw_model.config,
            "step":               step,
            "val_loss":           val_loss if (step % args.eval_interval == 0 or last_step) else None,
            "pretrain_step":      pretrain_step,
            "pretrain_val_loss":  pretrain_val_loss,
        }, ckpt_path)
        print(f"saved checkpoint → {ckpt_path}")

    # ── train step ────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        tok_np, msk_np = train_loader.next_batch()
        tok = torch.tensor(tok_np, dtype=torch.long,  device=device)
        msk = torch.tensor(msk_np, dtype=torch.float, device=device)

        x = tok[:, :-1]
        y = tok[:, 1:]
        m = msk[:, 1:]

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(x)

        loss = masked_loss(logits, y, m) / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    examples_per_sec = (args.micro_batch_size * grad_accum_steps) / (t1 - t0)
    print(
        f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} "
        f"| dt: {(t1-t0)*1000:.1f}ms | ex/s: {examples_per_sec:.1f}"
    )
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")

print("SFT training complete.")
