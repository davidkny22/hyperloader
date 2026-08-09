"""Public package surface for hyperloader."""

from ._hyperloader import package_version
from .config import AUTO, HyperConfig

__all__ = ["AUTO", "HyperConfig", "package_version"]
__version__ = package_version()
