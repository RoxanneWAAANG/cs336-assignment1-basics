import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: arbitrary
        # output shape: same as x

        # Find max value along the softmax dimension.
        # keepdim=True makes it broadcastable with x.
        x_max = x.max(dim=self.dim, keepdim=True).values

        # Subtract max for numerical stability.
        # softmax(x) == softmax(x - constant)
        x_shifted = x - x_max

        # Now exp will not overflow easily because the largest value is 0.
        x_exp = torch.exp(x_shifted)

        # Sum over the target softmax dimension.
        x_exp_sum = x_exp.sum(dim=self.dim, keepdim=True)

        # Normalize.
        softmax = x_exp / x_exp_sum

        return softmax