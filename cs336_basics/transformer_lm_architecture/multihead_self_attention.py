import torch
import torch.nn as nn
from einops import rearrange

from cs336_basics.transformer_lm_architecture.scaled_dot_product_attention import Attention
from cs336_basics.transformer_lm_architecture.rope import RoPE
from cs336_basics.transformer_lm_architecture.linear import Linear

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)

        if self.use_rope:
            assert self.head_dim % 2 == 0
            self.rope = RoPE(
                theta=rope_theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
            )
        else:
            self.rope = None

        self.attention = Attention(d_k=self.head_dim)
        self.o_proj = Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,  # (..., seq_len, d_model)
        M: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        seq_len = x.shape[-2]
        
        # 1. Project x into Q, K, V
        Q = self.q_proj(x)  # (..., seq_len, d_model)
        K = self.k_proj(x)  # (..., seq_len, d_model)
        V = self.v_proj(x)  # (..., seq_len, d_model)

        # 2. Split Q, K, V into heads
        Q = rearrange(
            Q,
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        K = rearrange(
            K,
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        V = rearrange(
            V,
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )

        # 3. Apply RoPE to Q and K, if enabled
        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        # 4. Adjust mask shape if needed
        if M is not None:
            if M.ndim == 2:
                M = rearrange(M, "seq_q seq_k -> 1 1 seq_q seq_k")
            elif M.ndim == 3:
                M = rearrange(M, "batch seq_q seq_k -> batch 1 seq_q seq_k")
        
        # 5. Apply scaled dot-product attention
        # x: (batch, num_heads, seq_len, head_dim)
        x = self.attention(M=M, Q=Q, K=K, V=V)

        # 6. Merge heads
        # x: (..., seq_len, d_model)
        x = rearrange(
            x,
            "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)",
        )

        # 7. Output projection
        x = self.o_proj(x)

        return x


