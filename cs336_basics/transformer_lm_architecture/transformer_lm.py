import torch
import torch.nn as nn

from cs336_basics.transformer_lm_architecture.linear import Linear
from cs336_basics.transformer_lm_architecture.embedding import Embedding
from cs336_basics.transformer_lm_architecture.rmsnorm import RMSNorm
# from cs336_basics.transformer_lm_architecture.softmax import Softmax
from cs336_basics.transformer_lm_architecture.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,  
    ):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
            )
            for _ in range(num_layers)
        ])

        self.final_layer_norm = RMSNorm(d_model)
        self.output_projection = Linear(d_model, vocab_size)
        # self.softmax = Softmax(dim=-1)
    
    def forward(self, x, M=None, token_positions=None):
        # token embedding.
        x = self.embedding(x)

        # transformer blocks.
        for block in self.transformer_blocks:
            x = block(x, M=M, token_positions=token_positions)

        # final layer norm.
        x = self.final_layer_norm(x)

        # output projection.
        x = self.output_projection(x)

        # # softmax activation.
        # x = self.softmax(x)
        
        # output logits.
        return x