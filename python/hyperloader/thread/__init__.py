"""Declared shared-address-space execution tier."""

from .iterator import ThreadIterator
from .pool import ThreadPool

__all__ = ["ThreadIterator", "ThreadPool"]
