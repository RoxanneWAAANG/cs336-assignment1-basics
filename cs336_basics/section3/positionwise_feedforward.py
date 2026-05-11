import torch
import torch.nn as nn
import math

from cs336_basics.section3.linear import Linear

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.linear3 = Linear(d_model, d_ff)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def glu(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.silu(x1) * x2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        # return: (batch_size, seq_len, d_model)
        x1 = self.linear1(x)  # (batch_size, seq_len, d_ff)
        x3 = self.linear3(x)  # (batch_size, seq_len, d_ff)
        x2 = self.glu(x1, x3)  # (batch_size, seq_len, d_ff)
        x = self.linear2(x2)   # (batch_size, seq_len, d_model)
        return x


