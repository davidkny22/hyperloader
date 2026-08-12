"""GPT-2-family models for recognizable pretraining anchors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class GptConfig:
    """GPT-2-family dimensions with tied token input and output weights."""

    name: str
    width: int
    depth: int
    attention_heads: int
    vocabulary_size: int = 50_257
    max_positions: int = 1_024

    def validate(self) -> None:
        """Reject dimensions that cannot form the named causal transformer."""
        dimensions = (
            self.width,
            self.depth,
            self.attention_heads,
            self.vocabulary_size,
            self.max_positions,
        )
        if not self.name or any(value <= 0 for value in dimensions):
            raise ValueError("GPT configuration fields must be positive")
        if self.width % self.attention_heads:
            raise ValueError("GPT width must be divisible by the attention-head count")

    def parameter_count(self) -> int:
        """Return the exact trainable parameter count for this implementation."""
        self.validate()
        embeddings = (self.vocabulary_size + self.max_positions) * self.width
        layer = 12 * self.width * self.width + 13 * self.width
        final_normalization = 2 * self.width
        return embeddings + self.depth * layer + final_normalization


GPT2_124M = GptConfig("GPT-2 124M", 768, 12, 12)
GPT2_355M = GptConfig("GPT-2 355M", 1_024, 24, 16)


class GptLanguageModel(nn.Module):
    """A pre-normalized causal GPT model with tied vocabulary weights."""

    def __init__(self, config: GptConfig) -> None:
        super().__init__()
        config.validate()
        self.vocabulary_size = config.vocabulary_size
        self.max_positions = config.max_positions
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.width)
        self.position_embedding = nn.Embedding(config.max_positions, config.width)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=config.width,
                nhead=config.attention_heads,
                dim_feedforward=4 * config.width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.depth)
        )
        self.normalization = nn.LayerNorm(config.width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return causal next-token logits for one integer token matrix."""
        sequence = int(tokens.shape[1])
        if sequence > self.max_positions:
            raise ValueError("token sequence exceeds the GPT position table")
        positions = torch.arange(sequence, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones((sequence, sequence), device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.layers:
            hidden = layer(hidden, src_mask=causal_mask, is_causal=True)
        hidden = self.normalization(hidden)
        return functional.linear(hidden, self.token_embedding.weight)
