# src/utils/misc.py
#!/usr/bin/env python3
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.
"""
General helpers:
  * set_seed  - reproducible runs
  * save_ckpt - atomic checkpoint write
"""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def save_ckpt(state: dict, path: str) -> None:
    temp = f"{path}.tmp"
    torch.save(state, temp)
    os.replace(temp, path)
