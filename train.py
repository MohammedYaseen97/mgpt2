import torch
import torch.nn.functional as F
import math
import os
import numpy as np
from model import GPT, GPTConfig

import argparse

## ------------------------------------------------------------
## Command line arguments
## ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--shards-dir",        default="data/shards_gpt2")
parser.add_argument("--seed",              type=int,   default=1337)
parser.add_argument("--total-batch-size",  type=int,   default=524288)
parser.add_argument("--micro-batch-size",  type=int,   default=16)
parser.add_argument("--max-lr",            type=float, default=3e-3)
parser.add_argument("--min-lr-ratio",      type=float, default=0.1)
parser.add_argument("--warmup-steps",      type=int,   default=715)
parser.add_argument("--max-steps",         type=int,   default=19073)
parser.add_argument("--weight-decay",      type=float, default=0.1)
parser.add_argument("--eval-interval",          type=int,   default=250)
parser.add_argument("--hellaswag-max-examples", type=int,   default=0,
                    help="Cap HellaSwag examples per eval (0 = all 10042)")
parser.add_argument("--log-dir",           default="logs/pretrain")
parser.add_argument("--tokenizer-kind",    default="gpt2",  choices=["gpt2", "mgpt2"])
parser.add_argument("--tokenizer-model",   default="tokenizer/artifacts/mgpt2.model")
args = parser.parse_args()

## ------------------------------------------------------------
## Data loading
## ------------------------------------------------------------
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        
        #get the shard filename
        data_root = args.shards_dir
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(self.shards) > 0, f"No shards found for split {split}"
        if master_process:
            print(f"loaded {len(self.shards)} shards for split {split}")
        self.reset()
        
    def reset(self):
        # state, init at shard 0
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = self.B * self.T * self.process_rank
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos:self.current_pos + B*T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_pos += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to the next shard
        if self.current_pos + (B * T * self.num_processes) + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = self.B * self.T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# simple launch:
# python mgpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=4 mgpt2.py

# run the training loop
import time
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken
from hellaswag import iterate_examples, render_example

# setup ddp
# torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # ddp atm requires cuda, we set the device appropriately according to the rank
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    # vanilla non-ddp run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to auto-detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

total_batch_size = args.total_batch_size
B = args.micro_batch_size
T = 1024  # block size — architecture constant
assert total_batch_size % (B * T * ddp_world_size) == 0, f"total_batch_size must be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

import types as _types
if args.tokenizer_kind == "gpt2":
    _e  = tiktoken.get_encoding("gpt2")
    enc = _types.SimpleNamespace(encode=_e.encode, decode=_e.decode)
else:
    from tokenizer.regex_tokenizer import RegexTokenizer as _RT
    _tok = _RT()
    _tok.load(args.tokenizer_model)
    enc = _types.SimpleNamespace(
        encode=lambda t: _tok.encode(t, allowed_special=set()),
        decode=_tok.decode,
    )
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
use_compile = False # torch compile intereferes with HellaSwag eval and Generation. TODO: fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr       = args.max_lr
min_lr       = max_lr * args.min_lr_ratio
warmup_steps = args.warmup_steps
max_steps    = args.max_steps
def get_lr(it):
    # 1. linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2. if it > max_steps, return min_lr
    if it > max_steps:
        return min_lr
    # 3. in between, use cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay, learning_rate=max_lr, device_type=device_type)

# create the log directory we will create checkpoints to and log to
log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    # once in a while, run validation loop
    if step % args.eval_interval == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps # normalize loss
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you wanted to more exactly resume training
                    print(f"saving checkpoint to {checkpoint_path}")
                    torch.save(checkpoint, checkpoint_path)
    
    # once in a while, evaluate HellaSwag
    # render_example receives enc so token IDs always match this model's vocab
    if (step % args.eval_interval == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        hella_cap = args.hellaswag_max_examples or 10_042
        for i, example in enumerate(iterate_examples("val")):
            if i >= hella_cap:
                break
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example, enc=enc)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
    
    # once in a while, generate from the model
    if ((step > 0 and step % args.eval_interval == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4 # Batch size: B
        max_length = 32 # Sequence length: T
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)  # Shape: (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :]  # Shape: (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)  # Shape: (B, vocab_size)
                # do topk sampling of 50 (huggingface pipeline default)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # Both shapes: (B, 50)
                # sample from the top 50 tokens
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # Shape: (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # Shape: (B, 1)
                # append the sampled tokens to the generated sequence
                xgen = torch.cat([xgen, xcol], dim=1)  # Shape: (B, T + 1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: ", decoded)
    
    # model training
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # used for forward pass also
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps # normalize loss
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set learning rate for this step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0 # difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tokens/s: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()