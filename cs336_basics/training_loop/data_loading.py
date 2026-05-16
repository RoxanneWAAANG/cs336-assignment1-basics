import torch
import numpy as np


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(x) - context_length
    if max_start <= 0:
        raise ValueError("dataset must be longer than context_length")

    start_indices = np.random.randint(0, max_start, size=batch_size)
    sequences = np.stack([x[i:i + context_length + 1] for i in start_indices])

    batch = torch.as_tensor(sequences, dtype=torch.long, device=device)
    return batch[:, :-1], batch[:, 1:]


class DataLoader:
    def __init__(
        self, x: np.ndarray,
        batch_size: int,
        context_length: int,
        device: str = 'mps'
    ):
        self.x = x
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return get_batch(
            x=self.x,
            batch_size=self.batch_size,
            context_length=self.context_length,
            device=self.device,
        )
