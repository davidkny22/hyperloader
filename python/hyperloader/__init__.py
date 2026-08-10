"""Public package surface for hyperloader."""

from ._hyperloader import package_version
from .api import DataLoader
from .config import AUTO, HyperConfig
from .rng import rng
from .stages import (
    Collate,
    Decode,
    Pipeline,
    Source,
    StageIO,
    ThreadSafety,
    Transform,
    pipeline,
)
from .verify import verify

__all__ = [
    "AUTO",
    "Collate",
    "DataLoader",
    "Decode",
    "HyperConfig",
    "Pipeline",
    "Source",
    "StageIO",
    "ThreadSafety",
    "Transform",
    "package_version",
    "pipeline",
    "rng",
    "verify",
]
__version__ = package_version()
