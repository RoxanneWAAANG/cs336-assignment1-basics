import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        # Store W, not W^T
        # Shape: (out_features, in_features)
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        # variance = 2 / (din + dout)
        # std = sqrt(variance)

        # Linear weights ~ N(0, 2 / (d_in + d_out))
        # truncated at [-3 sigma, 3 sigma]
        std = math.sqrt(2.0 / (self.in_features + self.out_features))

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., in_features)
        # self.weight shape: (out_features, in_features)
        # self.weight.T shape: (in_features, out_features)
        # output shape: (..., out_features)
        return x @ self.weight.T