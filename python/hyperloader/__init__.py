"""Public package surface for hyperloader."""

from ._hyperloader import package_version
from .api import DataLoader
from .config import AUTO, HyperConfig

__all__ = ["AUTO", "DataLoader", "HyperConfig", "package_version"]
__version__ = package_version()
