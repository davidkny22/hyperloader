"""Batch-native structured delivery."""

from .iterator import StructuredIterator
from .probe import is_native_batch_path, prepare_native_batch

__all__ = ["StructuredIterator", "is_native_batch_path", "prepare_native_batch"]
