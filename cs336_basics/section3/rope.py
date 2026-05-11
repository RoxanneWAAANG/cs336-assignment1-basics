import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(
        self, 
        theta: float, 
        d_k: int, 
        max_seq_len: int, 
        device=None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # check if d_k is even,
        # because RoPE requires the last dimension to be
        # split into pairs.
        assert d_k % 2 == 0, "d_k must be even for RoPE."

    def init_cache(self):
        # construct frequency for each pair of dimensions.
        # [0, 2, 4, ..., d_k - 2]
        k = torch.arange(0, self.d_k, 2, device=self.device).float()

        # inv_freq[j] = 1 / theta^(2j / d_k)
        # shape: (d_k / 2,)
        inv_freq = 1.0 / (self.theta ** (k / self.d_k))

        # construct angles for all possible token positions.
        # shape: (max_seq_len, d_k / 2)
        positions = torch.arange(self.max_seq_len, device=self.device).float()

        # angles[i, j] = position_i * inv_freq_j
        # shape:
        # positions[:, None] -> (max_seq_len, 1)
        # inv_freq[None, :] -> (1, d_k / 2)
        # result -> (max_seq_len, d_k / 2)
        angles = positions[:, None] * inv_freq[None, :]

        # precompute cos and sin.
        # shape: (max_seq_len, d_k / 2)
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)

        # register into buffer so that they will be moved
        # to the same device as the model.
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def calculate_rotation(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, d_k)
        # token_positions shape: (seq_len,)

        # get the precomputed cos and sin for given token positions.
        # cached shape: (max_seq_len, d_k / 2)
        # token_positions shape: (..., seq_len)
        # shape: (seq_len, d_k / 2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # split the last dimension into pairs.
        # x_even shape: (..., seq_len, d_k / 2)
        # [x0, x2, x4, ...]
        x_even = x[..., 0::2]

        # x_odd shape: (..., seq_len, d_k / 2)
        # [x1, x3, x5, ...]
        x_odd = x[..., 1::2]

        # apply rotation to each pair of dimensions.
        # [a'] = [a cos - b sin]
        # [b'] = [a sin + b cos]
        # rotated_even shape: (..., seq_len, d_k / 2)
        rotated_even = x_even * cos - x_odd * sin
        # rotated_odd shape: (..., seq_len, d_k / 2)
        rotated_odd = x_even * sin + x_odd * cos

        # interleave the rotated pairs back into the original shape.
        # shape: (..., seq_len, d_k / 2, 2)
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        # flatten the last two dimensions back to d_k.
        # (..., seq_len, d_k / 2, 2) -> (..., seq_len, d_k)
        rotated = rotated.flatten(start_dim=-2)

        return rotated
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, d_k)
        # token_positions shape: (..., seq_len)

        # ensure the last dimension is d_k.
        assert x.shape[-1] == self.d_k
        
        self.init_cache()
        result = self.calculate_rotation(x, token_positions)
        return result