import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = 1.0,
            a = -3,
            b = 3,
        )

    def forward (self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids shape: (...), values in [0, num_embeddings - 1]
        # self.weight shape: (num_embeddings, embedding_dim)
        # output shape: (..., embedding_dim)
        return self.weight[token_ids]