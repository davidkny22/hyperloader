"""Batch-native structured delivery."""

from .iterator import StructuredIterator
from .pipeline import bind_native_pipeline
from .probe import is_native_batch_path, prepare_native_batch

__all__ = [
    "StructuredIterator",
    "bind_native_pipeline",
    "is_native_batch_path",
    "prepare_native_batch",
]
