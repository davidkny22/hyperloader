"""Serialization of the three CPU-side global RNG states."""

from __future__ import annotations

import pickle
import random

import numpy as np
import torch

_NAMES = ("random", "numpy", "torch")


def capture_globals() -> dict[str, bytes]:
    """Serialize the ambient Python, NumPy, and torch CPU states."""
    return {
        "random": pickle.dumps(random.getstate(), protocol=pickle.HIGHEST_PROTOCOL),
        "numpy": pickle.dumps(np.random.get_state(), protocol=pickle.HIGHEST_PROTOCOL),
        "torch": _torch_state_bytes(torch.get_rng_state()),
    }


def restore_globals(payload: object) -> None:
    """Restore a validated ambient CPU RNG state bundle."""
    states = validate_globals(payload)
    random.setstate(pickle.loads(states["random"]))
    np.random.set_state(pickle.loads(states["numpy"]))
    torch.set_rng_state(_torch_state_tensor(states["torch"]))


def validate_globals(payload: object) -> dict[str, bytes]:
    """Validate one serialized CPU RNG state bundle."""
    if not isinstance(payload, dict) or set(payload) != set(_NAMES):
        raise TypeError("compat RNG state must contain random, numpy, and torch")
    states = {}
    for name in _NAMES:
        value = payload[name]
        if not isinstance(value, bytes):
            raise TypeError(f"compat {name} RNG state must be bytes")
        states[name] = value
    return states


def capture_generator(generator: object) -> bytes | None:
    """Serialize an explicit torch generator state when one exists."""
    if generator is None:
        return None
    return _torch_state_bytes(generator.get_state())


def restore_generator(generator: object, payload: bytes | None) -> None:
    """Restore an explicit torch generator state."""
    if payload is None:
        if generator is not None:
            raise ValueError("compat checkpoint generator presence changed")
        return
    if generator is None:
        raise ValueError("compat checkpoint requires its torch generator")
    generator.set_state(_torch_state_tensor(payload))


def _torch_state_bytes(state: torch.Tensor) -> bytes:
    """Return a storage-identity-free byte representation of one torch RNG state."""
    return state.detach().cpu().contiguous().numpy().tobytes()


def _torch_state_tensor(payload: bytes) -> torch.Tensor:
    """Reconstruct a writable CPU uint8 tensor from canonical RNG bytes."""
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
