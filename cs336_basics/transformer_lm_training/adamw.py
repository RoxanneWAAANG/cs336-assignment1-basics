from collections.abc import Callable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params, 
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=1e-2
        ):

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            ls = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Get state associated with p.
                state = self.state[p]

                # Get iteration number from the state, or initial value.
                t = state.get('t', 0) + 1

                # Get the gradient of loss with respect to p.
                grad = p.grad.data

                # Update the first moment estimate.
                m = state.get('m', torch.zeros_like(p.data))
                m = beta1 * m + (1 - beta1) * grad
                state['m'] = m

                # Update the second moment estimate.
                v = state.get('v', torch.zeros_like(p.data))
                v = beta2 * v + (1 - beta2) * grad * grad
                state['v'] = v

                # Computer adjusted learning for iteration t.
                lr_t = ls * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                
                # Update weight tensor in-place.
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                # Apply weight decay.
                p.data -= ls * weight_decay * p.data

        return loss