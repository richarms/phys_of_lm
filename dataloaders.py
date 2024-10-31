import torch
import numpy as np
import os
import random
from torch.utils.data import DataLoader

class DataLoaderFineWeb(DataLoader):
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
        # logging.info(f"found {len(shards)} shards for split {split}")
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

    def __len__(self):
        return len(self.tokens) // (self.B * self.T)
    
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