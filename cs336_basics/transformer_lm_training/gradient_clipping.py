import torch
from collections.abc import Iterable


class GradientClipper:
    def __init__(self, max_l2_norm: float, eps: float = 1e-6):
        if max_l2_norm <= 0:
            raise ValueError("max_l2_norm must be positive")

        self.max_l2_norm = max_l2_norm
        self.eps = eps

    def __call__(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Clip gradients of parameters in-place.

        Args:
            parameters: iterable of torch.nn.Parameter
            max_l2_norm: maximum allowed global gradient norm
        """

        params = [p for p in parameters if p.grad is not None]

        if len(params) == 0:
            return

        # Compute global L2 norm over all parameter gradients.
        total_norm_squared = torch.tensor(
            0.0,
            device=params[0].grad.device,
            dtype=params[0].grad.dtype,
        )

        for p in params:
            total_norm_squared += torch.sum(p.grad ** 2)

        total_norm = torch.sqrt(total_norm_squared)

        # If norm is too large, scale all gradients in-place.
        if total_norm > self.max_l2_norm:
            scale = self.max_l2_norm / (total_norm + self.eps)

            for p in params:
                p.grad.mul_(scale)