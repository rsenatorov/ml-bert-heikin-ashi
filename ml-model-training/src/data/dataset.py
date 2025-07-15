# src/data/dataset.py
#!/usr/bin/env python3
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.
"""
Dataset + collator for BERT-style masked LM pretraining on OHLC tokens.
"""

import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def _load_token_column(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if "Token" in df.columns:
        col = df["Token"]
    elif "token" in df.columns:
        col = df["token"]
    else:
        col = df.iloc[:,0]
    return col.to_numpy(dtype=np.int64)

class MaskedLanguageDataset(Dataset):
    def __init__(
        self,
        file_paths: list[str],
        context_size: int = 100,
        fraction: float = 1.0,
        mask_count: int = 15
    ):
        if not file_paths:
            raise ValueError("file_paths list is empty")
        self.ctx        = context_size
        self.mask_count = mask_count
        self.paths      = file_paths
        self.arrays     = []
        self.window_map = []

        for fidx, path in enumerate(self.paths):
            arr = _load_token_column(path)
            self.arrays.append(arr)
            n = len(arr)
            for start in range(n - context_size):
                self.window_map.append((fidx, start))

        if 0. < fraction < 1.:
            keep = max(1, int(len(self.window_map) * fraction))
            self.window_map = self.window_map[:keep]

    def __len__(self):
        return len(self.window_map)

    def __getitem__(self, idx):
        fidx, start = self.window_map[idx]
        arr = self.arrays[fidx]
        ctx = torch.as_tensor(
            arr[start:start + self.ctx],
            dtype=torch.long
        )
        return ctx

class BertMLMCollator:
    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        cls_token_id: int,
        mask_count: int = 15
    ):
        self.vocab_size    = vocab_size
        self.mask_token_id = mask_token_id
        self.cls_token_id  = cls_token_id
        self.mask_count    = mask_count

    def __call__(self, batch):
        input_ids = []
        labels    = []
        for ctx in batch:
            orig = ctx.tolist()
            seq  = [self.cls_token_id] + orig
            L    = len(seq)
            label = [-100] * L

            # pick exactly mask_count distinct positions (exclude CLS at 0)
            positions = random.sample(range(1, L), self.mask_count)
            mask_pos  = positions[:12]
            rand_pos  = positions[12:14]
            unchg_pos = positions[14:]

            for pos in mask_pos:
                label[pos] = seq[pos]
                seq[pos]   = self.mask_token_id

            for pos in rand_pos:
                label[pos] = seq[pos]
                seq[pos]   = random.randrange(self.vocab_size)

            for pos in unchg_pos:
                label[pos] = seq[pos]

            input_ids.append(seq)
            labels.append(label)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }
