"""Torch-shaped consumer reconstruction of detached worker exceptions."""

from __future__ import annotations

import importlib
import pickle
from typing import Any


class KeyErrorMessage(str):
    """String whose representation preserves formatted traceback text."""

    __slots__ = ()

    def __repr__(self) -> str:
        return self


def reraise_worker_exception(payload: bytes, worker: int) -> None:
    """Reconstruct torch's exception type and DataLoader traceback shape."""
    module_name, qualname, _original_message, formatted = pickle.loads(payload)
    message: str = (
        f"Caught {qualname} in DataLoader worker process {worker}.\n"
        f"Original {formatted}"
    )
    try:
        exception_type: Any = importlib.import_module(module_name)
        for component in qualname.split("."):
            exception_type = getattr(exception_type, component)
        if not isinstance(exception_type, type) or not issubclass(
            exception_type, BaseException
        ):
            raise TypeError
    except (AttributeError, ImportError, TypeError):
        raise RuntimeError(message) from None

    if exception_type is KeyError:
        message = KeyErrorMessage(message)
    elif getattr(exception_type, "message", None):
        raise exception_type(message=message)
    try:
        exception = exception_type(message)
    except Exception:
        raise RuntimeError(message) from None
    raise exception
