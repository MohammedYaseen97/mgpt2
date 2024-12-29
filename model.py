from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

@dataclass
class GPTConfig:
    block_size: int = 1024 # sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension

class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn= nn.Linear(config.n_embd, config.n_embd*3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(1,2)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(1,2)

        # att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
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
        x = x + self.attn(self.ln_1(x)) # (B, T, C)
        x = x + self.mlp(self.ln_2(x)) # (B, T, C)
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd), # token embedding table
            wpe=nn.Embedding(config.block_size, config.n_embd), # position embedding table
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer layers
            ln_f=nn.LayerNorm(config.n_embd), # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # language modeling head

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size() # (B, T) = batch size, sequence length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device = idx.device)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)

        for block in self.transformer.h: 
            x = block(x)

        x = self.transformer.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
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
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that are 2D ares going to be weight decayed.
        # i.e all weight tensors in matmul + embedding. All biases and layernorms are not.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        non_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': non_decay_params, 'weight_decay': 0.0}
        ]
        # num_decay_params = sum(p.numel() for p in decay_params)
        # num_non_decay_params = sum(p.numel() for p in non_decay_params)
        # if master_process:
        #     print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        #     print(f"num non-decayed parameter tensors: {len(non_decay_params)}, with {num_non_decay_params:,} parameters")
        # create AdamW optimizer and use fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        # if master_process:
        #     print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer