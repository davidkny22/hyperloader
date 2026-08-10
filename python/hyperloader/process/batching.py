"""Worker-side default collation for one native batch command."""

from __future__ import annotations

from typing import Any

from hyperloader import _hyperloader

from .serialization import ResultEncoder

BatchLayout = tuple[str, tuple[int, ...], int]
_NUMPY_ARRAY_TYPE: type[Any] | None = None


def _numpy_array_type() -> type[Any]:
    global _NUMPY_ARRAY_TYPE
    if _NUMPY_ARRAY_TYPE is None:
        import numpy as np

        _NUMPY_ARRAY_TYPE = np.ndarray
    return _NUMPY_ARRAY_TYPE


def batch_layout(value: Any) -> BatchLayout | None:
    """Describe an exact contiguous NumPy row eligible for in-place batching."""
    if (
        type(value) is not _numpy_array_type()
        or not value.flags.c_contiguous
        or value.dtype.hasobject
        or value.nbytes == 0
    ):
        return None
    try:
        import torch

        torch.from_numpy(value)
    except (TypeError, ValueError, RuntimeError):
        return None
    return value.dtype.str, tuple(value.shape), value.nbytes


def supports_worker_batch(value: Any) -> bool:
    """Return whether one probe selects worker-collated transport."""
    return batch_layout(value) is not None


def matches_batch_layout(value: Any, layout: BatchLayout) -> bool:
    """Return whether one produced row matches the probed raw layout exactly."""
    dtype, shape, row_bytes = layout
    return (
        type(value) is _numpy_array_type()
        and value.dtype.str == dtype
        and tuple(value.shape) == shape
        and value.nbytes == row_bytes
        and value.flags.c_contiguous
    )


def decode_batch(payload: Any, layout: BatchLayout) -> Any:
    """Wrap one arena-backed raw batch as its default-collated torch tensor."""
    import numpy as np
    import torch

    dtype, row_shape, row_bytes = layout
    view = memoryview(payload)
    if view.nbytes == 0 or view.nbytes % row_bytes:
        raise RuntimeError("Worker batch slot has an invalid byte length.")
    rows = view.nbytes // row_bytes
    torch_dtype = torch.from_numpy(np.empty(0, dtype=np.dtype(dtype))).dtype
    return torch.frombuffer(view, dtype=torch_dtype).reshape((rows, *row_shape))


def encode_batch(values: list[Any], encoder: ResultEncoder) -> bytes:
    """Collate one default-collation batch and encode its storage once."""
    batch = _hyperloader._default_collate(values)
    return encoder.encode_uncached(batch)
