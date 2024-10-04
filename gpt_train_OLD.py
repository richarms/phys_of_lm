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
# import transformers

# -----------------------------------------------------------------------------
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO) 

enc = tiktoken.get_encoding("gpt2")

# use cpu or gpu based on your system
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
logging.info(f'Using device: {device}')
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.set_float32_matmul_precision('medium')

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
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
    block_size: int = 1024 # max sequence length (T)
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-rich', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        logging.info(f'loading weights from pretrained gpt: {model_type}')

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-rich':    dict(n_layer=16, n_head=16, n_embd=768),  # ?M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
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

class DataLoader:
    def __init__(self, data, encoder, batch_size, seq_len):
        self.B = batch_size
        self.T = seq_len

        # load the data into memory
        with open(data) as f:
            self.data = f.read()
        self.tokens = torch.tensor(encoder.encode(self.data))
        logging.info(f'loaded {len(self.tokens)} tokens')
        logging.info(f'1 epoch = {len(self.tokens) // (self.B * self.T)} batches')

        # state
        self.current_idx = 0
    
    def __iter__(self):
        B, T = self.B, self.T
        while 1:
            if self.current_idx >= len(self.tokens) - (B*T + 1):
                self.current_idx = 0
            # get the next batch
            buf = self.tokens[self.current_idx:self.current_idx + B*T + 1]
            x = buf[:-1].view(B, T)
            y = buf[1:].view(B, T)
            self.current_idx += B*T
            yield x, y


model = GPT.from_pretrained('gpt2')
# model = GPT(GPTConfig(vocab_size=50304))

model.eval()
model.to(device)
model = torch.compile(model)

B: int = 16 # micro_batch_size
total_batch_size: int = 524288 # = 512 * GPTConfig.block_size 
assert total_batch_size % (B * GPTConfig.block_size) == 0
grad_accum_steps: int = total_batch_size // (B * GPTConfig.block_size)

min_lr:float = 6e-5
max_lr:float = 6e-4
max_steps: int = 19073 #100
warmup_steps: int = 715 #5

optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=1e-8, lr=max_lr, fused=True)

BioS = 'synthetic_augmented_biographies.txt'
# train_loader = DataLoader(data='input.txt', encoder=enc, batch_size=B, seq_len=GPTConfig.block_size)
train_loader = DataLoader(data=BioS, encoder=enc, batch_size=B, seq_len=GPTConfig.block_size)
bio_loader = DataLoaderFineWeb(B, GPTConfig.block_size, 'train')
# always validate on fineweb
val_loader = DataLoaderFineWeb(B, GPTConfig.block_size, 'val')

def get_lr(step=0):
    # linear warmup
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * step / warmup_steps
    # cosine decay
    elif step < max_steps:
        return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(step * math.pi / max_steps))
    else:
        return min_lr


# optimise
for step in range(1,max_steps):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        # 10% of the time, load data from the 2nd dataset
        x, y = next(iter(train_loader)) if random.random() > 0.1 else next(iter(bio_loader))
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, targets=y)
            # import code; code.interact(local=locals())
        loss = loss / grad_accum_steps 
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for p in optimizer.param_groups:
        p['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    tokens_per_second = (grad_accum_steps * B * GPTConfig.block_size) / dt
    logging.info(f'step: {step} | loss: {loss_accum.item():.3f} | time: {dt*1000:.2f}ms | tok/s: {tokens_per_second} | norm: {norm:.4f} | lr: {lr:.5f}')


# once in a while evaluate our validation loss
    if step % 250 == 0 or step == max_steps - 1:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = next(iter(val_loader))
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        logging.info(f"validation loss: {val_loss_accum.item():.4f}")
    # save the model
    # if step % 1000 == 0 or step == max_steps - 1:
    #    model.save_checkpoint(f'checkpoint_{step}.pt')

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or (step == max_steps - 1)) :#and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("One person who has been on my mind of late is ")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"sample {i}: {decoded}")

# import sys; sys.exit(0)

# test generation
num_return_sequences = 10
max_length = 48
tokens = enc.encode("One person who has been on my mind of late is ")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)
#print(f'xgen shape:', xgen.shape)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42)
while xgen.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad(): # saying to torch that do not store gradients for whatever we do below
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(xgen) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        xgen = torch.cat((xgen, xcol), dim=1)
# print the generated text
for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"sample {i}: {decoded}")