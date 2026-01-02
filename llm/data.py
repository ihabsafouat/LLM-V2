# llm/data.py
from __future__ import annotations
import torch
from typing import Tuple

class TokenDataset:
    """
    Holds the full tokenized corpus in one tensor and samples random
    (x, y) batches for causal language modeling.
    """
    def __init__(self, token_ids: torch.Tensor, block_size: int, device: str = "cpu"):
        assert token_ids.dim() == 1, "token_ids must be a 1D tensor"
        assert token_ids.numel() > block_size + 1, "Not enough tokens for block_size"
        self.data = token_ids.to(torch.long)
        self.block_size = block_size
        self.device = device

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # random starting indices
        max_i = self.data.numel() - self.block_size - 1
        ix = torch.randint(0, max_i, (batch_size,))

        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix])

        return x.to(self.device), y.to(self.device)
