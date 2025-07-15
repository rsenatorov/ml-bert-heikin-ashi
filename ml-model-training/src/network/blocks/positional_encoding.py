# src/network/blocks/positional_encoding.py
#!/usr/bin/env python3
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.
"""
Learned positional embeddings.
"""

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, context: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, context, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos[:, : x.size(1), :]
