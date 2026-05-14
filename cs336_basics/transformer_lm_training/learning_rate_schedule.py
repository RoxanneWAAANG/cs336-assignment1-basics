import math
import torch
from collections.abc import Callable
from typing import Optional


class CosineSchedule:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_min: float,
        lr_max: float,
        warmup_steps: int,
        total_steps: int,
    ):
        self.optimizer = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.t = 0

    def get_lr(self):
        if self.t < self.warmup_steps:
            return self.lr_max * self.t / self.warmup_steps

        if self.t > self.total_steps:
            return self.lr_min

        progress = (self.t - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )

        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + math.cos(math.pi * progress)
        )

        return lr

    def step(self):
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.t += 1