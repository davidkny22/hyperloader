"""Continuous compute- and bandwidth-bound GPU work for paired cells."""

from __future__ import annotations

from typing import Any


class GpuWorkload:
    """One warmed GPU operation consuming each delivered input batch."""

    def __init__(self, regime: str) -> None:
        import torch

        if regime not in {"compute", "bandwidth"}:
            raise ValueError("GPU regime must be compute or bandwidth")
        self._torch = torch
        self.regime = regime
        self._batch_checksum: Any = None
        if regime == "compute":
            self._left = torch.randn((4096, 4096), dtype=torch.float16, device="cuda")
            self._right = torch.randn((4096, 4096), dtype=torch.float16, device="cuda")
            self._output = torch.empty_like(self._left)
        else:
            self._source = torch.randn(64 * 1024 * 1024, device="cuda")
            self._output = torch.empty_like(self._source)

    def run(self, batch: Any) -> None:
        """Transfer one batch, execute one fixed GPU operation, and synchronize."""
        device_batch = batch.to("cuda", non_blocking=False)
        self._batch_checksum = device_batch.sum()
        if self.regime == "compute":
            self._torch.mm(self._left, self._right, out=self._output)
        else:
            self._torch.mul(self._source, 1.0001, out=self._output)
        self._torch.cuda.synchronize()

    def warm(self, batch: Any, iterations: int = 20) -> None:
        """Warm kernels, allocations, and clocks outside timing."""
        for _ in range(iterations):
            self.run(batch)
