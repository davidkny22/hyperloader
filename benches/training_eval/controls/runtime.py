"""Training-library and allocator-state capture."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from typing import Any


def torch_threading() -> dict[str, Any]:
    """Return consumer-side Torch threading settings."""
    try:
        import torch

        return {
            "consumer_inter_op_threads": torch.get_num_interop_threads(),
            "consumer_intra_op_threads": torch.get_num_threads(),
        }
    except (ImportError, RuntimeError) as error:
        return {"unavailable": f"{type(error).__name__}: {error}"}


def allocator_state() -> dict[str, Any]:
    """Return CUDA allocator configuration and live totals."""
    values: dict[str, Any] = {
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    try:
        import torch

        if not torch.cuda.is_available():
            values["cuda"] = "unavailable"
            return values
        values.update(
            {
                "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        backend = getattr(torch.cuda.memory, "get_allocator_backend", None)
        values["cuda_allocator_backend"] = (
            backend() if backend is not None else "unavailable"
        )
    except (ImportError, RuntimeError) as error:
        values["cuda"] = f"{type(error).__name__}: {error}"
    return values


def library_versions() -> dict[str, str]:
    """Return installed identities for every participating training library."""
    packages = (
        "hyperloader",
        "numpy",
        "pillow",
        "spdl",
        "torch",
        "torchvision",
    )
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions
