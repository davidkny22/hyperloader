"""Real transformer language model used by the synthetic step-time dial."""

from __future__ import annotations

import torch
from torch import nn

from .dial import TransformerDialPoint


class DialTransformer(nn.Module):
    """A next-token transformer whose architecture is controlled by one dial point."""

    def __init__(self, point: TransformerDialPoint) -> None:
        super().__init__()
        point.validate()
        self.vocabulary_size = point.vocabulary_size
        self.embedding = nn.Embedding(point.vocabulary_size, point.width)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=point.width,
                nhead=point.attention_heads,
                dim_feedforward=4 * point.width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(point.depth)
        )
        self.normalization = nn.LayerNorm(point.width)
        self.output = nn.Linear(point.width, point.vocabulary_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return token logits for one integer token matrix."""
        hidden = self.embedding(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.output(self.normalization(hidden))
