"""Forward, backward, optimizer, and device-transfer execution for the dial."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.nn import functional

from .feeders import TokenBatch
from .transformer import DialTransformer


class TransformerStepRunner:
    """Keep one model and optimizer alive across all feeder halves."""

    def __init__(
        self,
        model: DialTransformer,
        *,
        device: torch.device,
        precision: str,
        learning_rate: float,
        non_blocking: bool,
    ) -> None:
        if precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("precision must be float32, float16, or bfloat16")
        if precision == "float16" and device.type != "cuda":
            raise ValueError("float16 dial execution requires CUDA")
        self.model = model.to(device)
        self.device = device
        self.precision = precision
        self.non_blocking = non_blocking
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self._scaler = torch.amp.GradScaler(
            "cuda", enabled=precision == "float16"
        )

    def step(self, batch: TokenBatch) -> torch.Tensor:
        """Execute one real next-token forward, backward, and optimizer step."""
        tokens = batch.tokens.to(self.device, non_blocking=self.non_blocking)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            logits = self.model(inputs)
            loss = functional.cross_entropy(
                logits.reshape(-1, self.model.vocabulary_size),
                targets.reshape(-1),
            )
        self._scaler.scale(loss).backward()
        self._scaler.step(self.optimizer)
        self._scaler.update()
        return loss.detach()

    def finish(self, loss: torch.Tensor) -> float:
        """Synchronize outstanding device work and return the terminal scalar loss."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return float(loss.float().item())

    def _autocast(self) -> Any:
        if self.precision == "float32":
            return nullcontext()
        dtype = torch.float16 if self.precision == "float16" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype)
