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

from gpt_train_refactor import GPT, GPTConfig
from transformers import GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType

# Device configuration
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
logging.info(f'Using device: {device}')
device_type = "cuda" if device.startswith("cuda") else "cpu"
torch.set_float32_matmul_precision('medium')

model = GPT.from_pretrained('gpt2')

# load checkpoint
#PATH = 'checkpoints/model_00010.pt'
#checkpoint = torch.load(PATH, weights_only=False)

import sys; sys.exit()

model.load_state_dict(checkpoint['model'])

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

