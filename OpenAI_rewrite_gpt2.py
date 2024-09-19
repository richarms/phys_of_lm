import os
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers
import tiktoken

# Setup device
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.set_float32_matmul_precision('high')

# Encoder setup
enc = tiktoken.get_encoding("gpt2")

# -----------------------------------------------------------------------------
# Model Components

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

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

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config = GPTConfig(**config_args, vocab_size=50257, block_size=1024)
        return GPT(config)

class DataLoader:
    def __init__(self, data, encoder, batch_size, seq_len):
        self.B = batch_size
        self.T = seq_len
        with open(data) as f:
            self.data = f.read()
        self.tokens = torch.tensor(encoder.encode(self.data))
        self.current_idx = 0

    def __iter__(self):
        while True:
            if self.current_idx >= len(self.tokens) - (self.B * self.T + 1):
                self.current_idx = 0
            buf = self.tokens[self.current_idx:self.current_idx + self.B * self.T + 1]
            x = buf[:-1].view(self.B, self.T)
            y = buf[1:].view(self.B, self.T)
            self.current_idx += self.B * self.T
            yield x, y

# -----------------------------------------------------------------------------
# Training Setup

B = 16
total_batch_size = 524288
grad_accum_steps = total_batch_size // (B * GPTConfig.block_size)

min_lr = 6e-5
max_lr = 6e-4
max_steps = 19073
warmup_steps = 715

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304))
model.eval()
model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=1e-8, lr=max_lr)
train_loader = DataLoader(data='input.txt', encoder=enc, batch_size=B, seq_len=GPTConfig.block_size)

def get_lr(step):
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * step / warmup_steps
    elif step < max_steps:
        return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(step * math.pi / max_steps))
    else:
        return min_lr

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for _ in range(grad_accum_steps):
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, targets=y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for p in optimizer.param_groups:
        p['lr'] = lr
    optimizer.step()

    dt = time.time() - t0
    tokens_per_second = (grad_accum_steps * B * GPTConfig.block_size) / dt
    print(f'step: {step} | loss: {loss_accum.item():.3f} | time: {dt*1000:.2f}ms | tok/s: {tokens_per_second:.2f} | lr: {lr:.5f}')

# -----------------------------------------------------------------------------
# Generation Example

num_return_sequences = 4
max_length = 32
tokens = enc.encode("Who shall ")
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(device)

sample_rng = torch.Generator(device=device).manual_seed(42)
while tokens.size(1) < max_length:
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(tokens)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
        next_token = torch.gather(topk_indices, -1, ix)
        tokens = torch.cat([tokens, next_token], dim=1)

for i in range(num_return_sequences):
    decoded = enc.decode(tokens[i, :max_length].tolist())
    print(f"Sample {i}: {decoded}")
