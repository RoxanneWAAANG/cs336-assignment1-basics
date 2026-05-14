import torch
import torch.nn as nn
from cs336_basics.transformer_lm_architecture.softmax import Softmax

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits shape: (batch_size, vocab_size)
        # targets shape: (batch_size,)

        batch_size, _ = logits.shape

        # correct logits: o_i[y_i]
        correct_logits = logits[torch.arange(batch_size), targets]

        # logsumexp: log(sum_j exp(o_i[j])))
        log_sum_exp = torch.logsumexp(logits, dim=-1)

        # cross entropy for each example
        loss = log_sum_exp - correct_logits

        # average over batch
        return loss.mean()
    
class Perplexity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits shape: (batch_size, vocab_size)
        # targets shape: (batch_size,)

        batch_size, _ = logits.shape

        # correct logits: o_i[y_i]
        correct_logits = logits[torch.arange(batch_size), targets]

        # logsumexp: log(sum_j exp(o_i[j])))
        log_sum_exp = torch.logsumexp(logits, dim=-1)

        # cross entropy for each example
        cross_entropy = log_sum_exp - correct_logits

        # average over batch
        avg_cross_entropy = cross_entropy.mean()

        # perplexity is exp of average cross entropy
        perplexity = torch.exp(avg_cross_entropy)

        return perplexity
