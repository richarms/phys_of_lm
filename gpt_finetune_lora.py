from dataclasses import dataclass
import logging
import math
# import numpy as np
import random
import tiktoken
import time
import torch

from torch.nn import functional as F

from dataloaders import DataLoader, DataLoaderFineWeb
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer, default_data_collator
from peft import get_peft_model, LoraConfig, TaskType

# Device configuration
def infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = infer_device()

device_type = "cuda" if device.startswith("cuda") else "cpu"
torch.set_float32_matmul_precision('medium')

# # load checkpoint
# PATH = 'checkpoints/model_00010.pt'
# checkpoint = torch.load(PATH, weights_only=False)
# state_dict = checkpoint['model']
#  # fix the keys of the state dictionary
# unwanted_prefix = '_orig_mod.' 
# for k,v in list(state_dict.items()): 
#     if k.startswith(unwanted_prefix): 
#         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k) 

# model.load_state_dict(state_dict)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, 
    r=64, 
    lora_alpha=128, 
    lora_dropout=0.0,
    target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"]
)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

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



# Model initialization
def initialize_model():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.eval()
    model.to(device)
    model = torch.compile(model)
    return model

model = initialize_model()
optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=1e-8, lr=MAX_LR, fused=True)

# Token encoder
enc = tiktoken.get_encoding("gpt2")


if __name__ == "__main__":
    # Data Loaders
    BioS = 'synthetic_augmented_biographies.txt'
    with open(BioS) as f:
            tokens = torch.tensor(enc.encode(f.read()))
    bio_loader = DataLoader(tokens, batch_size=BATCH_SIZE)
    # train_loader = DataLoaderFineWeb(BATCH_SIZE, GPTConfig.block_size, 'train')
    # val_loader = DataLoaderFineWeb(BATCH_SIZE, GPTConfig.block_size, 'val')

    training_args = TrainingArguments(
        output_dir="lora_checkpoints",
        num_train_epochs=2,
        save_total_limit=5,
        per_device_train_batch_size=8,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.0001,
        dataloader_drop_last=True,
        bf16=True,
        logging_steps=10,
        learning_rate=1e-5,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        hub_model_id="gpt2-lora-bio-fine-tuned",
        max_steps=MAX_STEPS,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=bio_loader,
        data_collator=default_data_collator,
    )
    # model.config.use_cache = False
    trainer.train()


