from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("./tokenizer/mgpt2-tokenizer")

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn= nn.Linear(config.n_embd, config.n_embd*3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(1,2)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(1,2)

        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = nn.Softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B,T,C)

        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    @classmethod
    def from_pretrained(cls, model_type
                        ):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt {model_type}..")

        config_args = {
            "gpt2":         dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium":  dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large":   dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":      dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        transposed_keys = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"Mismatch: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(suffix) for suffix in transposed_keys):
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
        
        