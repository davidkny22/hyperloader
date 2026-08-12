"""Forward, backward, and optimizer execution for vision anchors."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from .image_batches import ImageBatch


class VisionStepRunner:
    """Keep one vision model and AdamW optimizer alive across feeder halves."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device,
        precision: str,
        learning_rate: float,
        non_blocking: bool,
    ) -> None:
        if precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("precision must be float32, float16, or bfloat16")
        if precision == "float16" and device.type != "cuda":
            raise ValueError("float16 vision execution requires CUDA")
        self.model = model.to(device)
        self.device = device
        self.precision = precision
        self.non_blocking = non_blocking
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self._scaler = torch.amp.GradScaler(
            "cuda", enabled=precision == "float16"
        )

    def step(self, batch: ImageBatch) -> torch.Tensor:
        """Execute one real image-classification optimizer step."""
        batch.validate()
        images = batch.images.to(self.device, non_blocking=self.non_blocking)
        labels = batch.labels.to(self.device, non_blocking=self.non_blocking)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            loss = functional.cross_entropy(self.model(images), labels)
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
