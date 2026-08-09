"""Platform input and direct-I/O configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class IOConfig:
    """Configure platform I/O selection and direct-I/O policy."""

    backend: Literal["auto", "uring", "iocp", "pread"] = "auto"
    direct: Literal["auto", "on", "off"] = "auto"
