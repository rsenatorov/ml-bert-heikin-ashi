# src/utils/scheduler.py
#!/usr/bin/env python3
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.
"""
Cosine decay with linear warm-up.
One scheduler step equals one batch.
"""

import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps:    int,
        eta_min:    float = 0.0,
        last_epoch:   int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self.eta_min      = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
            return [base * scale for base in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.max_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base - self.eta_min) * cosine
            for base in self.base_lrs
        ]
