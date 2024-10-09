from dataclasses import dataclass
import logging
import math
import numpy as np
import os
import random
import tiktoken
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Device configuration
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
logging.info(f'Using device: {device}')
device_type = "cuda" if device.startswith("cuda") else "cpu"
torch.set_float32_matmul_precision('medium')

# Token encoder
enc = tiktoken.get_encoding("gpt2")

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
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
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
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
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel
        assert model_type in {'gpt2', 'gpt2-rich', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        logging.info(f'loading weights from pretrained gpt: {model_type}')
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-rich': dict(n_layer=16, n_head=16, n_embd=768),  # ?M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


class DataLoader:
    def __init__(self, data, encoder, batch_size, seq_len):
        self.B = batch_size
        self.T = seq_len
        with open(data) as f:
            self.tokens = torch.tensor(encoder.encode(f.read()))
        self.current_idx = 0
    
    def __iter__(self):
        B, T = self.B, self.T
        while True:
            if self.current_idx >= len(self.tokens) - (B * T + 1):
                self.current_idx = 0
            buf = self.tokens[self.current_idx:self.current_idx + B * T + 1]
            x = buf[:-1].view(B, T)
            y = buf[1:].view(B, T)
            self.current_idx += B * T
            yield x, y

class DataLoaderFineWeb:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        logging.info(f"found {len(shards)} shards for split {split}")
        self.reset()
    
    def load_tokens(self, filename):
        npt = np.load(filename).astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def reset(self):
        # Shuffle the shards at the start of each epoch
        random.shuffle(self.shards)
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = 0  # Reset position

    def __iter__(self, num_batches=None):
        num_batches = num_batches or float('inf')  # Iterate infinitely unless limited
        batch_count = 0
        B, T = self.B, self.T

        while batch_count < num_batches:
            buf = self.tokens[self.current_position : self.current_position + B * T + 1]
            x = buf[:-1].view(B, T)  # inputs
            y = buf[1:].view(B, T)  # targets
            
            # advance the position in the tensor
            self.current_position += B * T

            # if loading the next batch would be out of bounds, advance to next shard
            if self.current_position + (B * T + 1) > len(self.tokens):
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = self.load_tokens(self.shards[self.current_shard])
                self.current_position = 0  # Reset position to the start of the new shard

            yield x, y
            batch_count += 1

# Configuration for training
BATCH_SIZE = 16  # Micro batch size
TOTAL_BATCH_SIZE = 524288  # = 512 * GPTConfig.block_size
MAX_STEPS = 19073
WARMUP_STEPS = 715
CHECKPOINT_EVERY = 10
MIN_LR = 6e-5
MAX_LR = 6e-4

assert TOTAL_BATCH_SIZE % (BATCH_SIZE * GPTConfig.block_size) == 0      # Verify batch size divisibility
grad_accum_steps = TOTAL_BATCH_SIZE // (BATCH_SIZE * GPTConfig.block_size)

# Learning rate scheduler
def get_lr(step):
    if step < WARMUP_STEPS:
        return MIN_LR + (MAX_LR - MIN_LR) * step / WARMUP_STEPS
    elif step < MAX_STEPS:
        return MIN_LR + (MAX_LR - MIN_LR) * 0.5 * (1.0 + math.cos(step * math.pi / MAX_STEPS))
    else:
        return MIN_LR

# Model initialization
def initialize_model():
    model = GPT.from_pretrained('gpt2')
    # model = GPT(GPTConfig(vocab_size=50304))
    model.eval()
    model.to(device)
    model = torch.compile(model)
    return model

model = initialize_model()
optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=1e-8, lr=MAX_LR, fused=True)

# Training loop
def train_model(model, train_loader, bio_loader, val_loader):
    for step in range(MAX_STEPS):
        start_time = time.time()
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        for _ in range(grad_accum_steps):
            # Choose between fineweb and bios data with 0.1 prob
            x, y = next(iter(train_loader)) if random.random() > 0.1 else next(iter(bio_loader))
            # x, y = next(iter(bio_loader))
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, targets=y)

            loss = loss / grad_accum_steps
            total_loss += loss.detach()
            loss.backward()

        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        # Timing, throughput, and logging
        elapsed_time = time.time() - start_time
        tokens_per_second = (grad_accum_steps * BATCH_SIZE * GPTConfig.block_size) / elapsed_time
        logging.info(
            f"Step: {step} | Loss: {total_loss.item():.3f} | Time: {elapsed_time:.2f}s | "
            f"Tokens/s: {tokens_per_second:.2f} | LR: {lr:.5f}"
        )

        # Validation 
        if step % 250 == 0 or step == MAX_STEPS - 1:
            val_loss = validate_model(model, val_loader)

        # Model checkpoints 
        if step > 0 and (step % CHECKPOINT_EVERY == 0 or step == MAX_STEPS - 1):
                checkpoint_path = os.path.join('./checkpoints/', f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': val_loss.item()
                }
                # add optimizer.state_dict(), seeds, etc. if more exact training resumption required
                torch.save(checkpoint, checkpoint_path)

# Validation
def validate_model(model, val_loader):
    model.eval()
    val_loader.reset()
    total_val_loss = 0.0
    val_loss_steps = 20

    with torch.no_grad():
        for _ in range(val_loss_steps):
            x, y = next(iter(val_loader))
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            total_val_loss += loss.detach() / val_loss_steps

    logging.info(f"Validation Loss: {total_val_loss.item():.4f}")
    return total_val_loss

# Sampling function
def generate_samples(model, prompt="I am a language model and ", num_sequences=4, max_length=32):
    model.eval()
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).repeat(num_sequences, 1).to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    while tokens.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            tokens = torch.cat((tokens, torch.gather(topk_indices, -1, ix)), dim=1)

    for i in range(num_sequences):
        decoded = enc.decode(tokens[i, :max_length].tolist())
        print(f"Sample {i}: {decoded}")

if __name__ == "__main__":
    # Data Loaders
    BioS = 'synthetic_augmented_biographies.txt'
    train_loader = DataLoader(data=BioS, encoder=enc, batch_size=BATCH_SIZE, seq_len=GPTConfig.block_size)
    bio_loader = DataLoaderFineWeb(BATCH_SIZE, GPTConfig.block_size, 'train')
    val_loader = DataLoaderFineWeb(BATCH_SIZE, GPTConfig.block_size, 'val')

    train_model(model, train_loader, bio_loader, val_loader)

    generate_samples(model)
