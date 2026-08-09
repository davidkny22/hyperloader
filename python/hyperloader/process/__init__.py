"""Persistent black-box process execution."""

from .pool import ProcessPool
from .seed import resolve_root_seed

__all__ = ["ProcessPool", "resolve_root_seed"]
