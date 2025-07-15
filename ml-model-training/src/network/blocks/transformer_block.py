# src/network/blocks/transformer_block.py
#!/usr/bin/env python3
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.
"""
Transformer encoder block (no causal masking).
"""

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff:    int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        y, _ = self.attn(
            self.ln1(x), self.ln1(x), self.ln1(x)
        )
        x = x + self.drop(y)
        x = x + self.ff(self.ln2(x))
        return x
