"""Process-worker library environment configuration."""

from __future__ import annotations


def configure_worker_environment() -> None:
    """Match Torch's intra-op isolation before loading dataset code."""
    import torch

    torch.set_num_threads(1)
