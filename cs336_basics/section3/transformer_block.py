import torch
import torch.nn as nn

from cs336_basics.section3.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.section3.rmsnorm import RMSNorm
from cs336_basics.section3.positionwise_feedforward import FFN

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: float = 0.0,
            max_seq_len: int = 2048,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True if theta > 0 else False,
            rope_theta=theta,
            max_seq_len=max_seq_len,
        )

        self.ffn = FFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, M: torch.Tensor | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x_norm1 = self.norm1(x)
        x_attn = self.self_attention(x_norm1, M=M, token_positions=token_positions)
        
        x = x + x_attn

        x_norm2 = self.norm2(x)
        x_ffn = self.ffn(x_norm2)
        x = x + x_ffn
        
        return x