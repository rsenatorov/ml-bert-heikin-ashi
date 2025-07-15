# src/network/model.py
#!/usr/bin/env python3
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.
"""
BERT-style Transformer encoder for masked LM on OHLC tokens.
"""

import torch
import torch.nn as nn
from network.blocks.positional_encoding import PositionalEncoding
from network.blocks.transformer_block    import TransformerBlock

class BertTimeSeries(nn.Module):
    def __init__(
        self,
        orig_vocab_size: int = 1536,
        context_size:    int = 101,
        d_model:         int = 768,
        n_heads:         int = 12,
        d_ff:            int = 3072,
        n_layers:        int = 12,
        dropout:       float = 0.1,
    ):
        super().__init__()

        self.orig_vocab_size   = orig_vocab_size
        self.cls_token_id      = orig_vocab_size
        self.mask_token_id     = orig_vocab_size + 1
        self.extended_vocab_sz = orig_vocab_size + 2

        self.token_emb = nn.Embedding(self.extended_vocab_sz, d_model)
        self.pos_enc   = PositionalEncoding(context_size, d_model)
        self.drop      = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        # MLM head: map back to original vocab size
        self.head = nn.Linear(d_model, orig_vocab_size, bias=True)
        self.context_size = context_size

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        # input_ids: (B, seq_len==context_size)
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)  # no causal mask

        x = self.ln_f(x)
        # logits for each position
        return self.head(x)  # (B, seq_len, orig_vocab_size)
