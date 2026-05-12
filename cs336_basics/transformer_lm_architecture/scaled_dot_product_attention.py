import torch
import torch.nn as nn
from cs336_basics.transformer_lm_architecture.softmax import Softmax

class Attention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        M: torch.Tensor,
        Q: torch.Tensor,  # shape: (..., sequence_length_q, d_k)
        K: torch.Tensor,  # shape: (..., sequence_length_k, d_k)
        V: torch.Tensor,  # shape: (..., sequence_length_v, d_v), where sequence_length_v == sequence_length_k
    ) -> torch.Tensor:
        # output shape: (..., sequence_length_q, d_v)

        # Compute scaled dot product attention.
        # Step 1: Compute raw attention scores by taking the dot product of Q and K^T.
        # The resulting shape will be (..., sequence_length_q, sequence_length_k).
        if M is not None:
            # apply mask M to attention scores by setting masked positions to -inf.
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            attention_scores = attention_scores.masked_fill(M == 0, float("-inf"))
        else:
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Step 2: Apply softmax to get attention weights.
        # The softmax is applied along the last dimension (sequence_length_k).
        attention_weights = Softmax(dim=-1)(attention_scores)

        # Step 3: Use the attention weights to compute a weighted sum of the values V.
        # The resulting shape will be (..., sequence_length_q, d_v).
        output = torch.matmul(attention_weights, V)

        return output