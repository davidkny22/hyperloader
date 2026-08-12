"""Pure-Python execution primitives used when the native module is absent."""

from .native import IS_FALLBACK

__all__ = ["IS_FALLBACK"]
