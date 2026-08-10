"""Public package surface for hyperloader."""

from ._hyperloader import package_version
from .api import DataLoader
from .config import AUTO, HyperConfig
from .rng import rng
from .verify import verify

__all__ = ["AUTO", "DataLoader", "HyperConfig", "package_version", "rng", "verify"]
__version__ = package_version()
