"""Environment metadata capture for benchmark records."""

from __future__ import annotations

import platform
import socket
import sysconfig
from datetime import datetime, timezone

from .models import EnvironmentMetadata


def capture_environment(
    *,
    commit: str,
    cpu_governor: str,
    gpu_clock: str,
    cache_regime: str,
    benchmark_mode: bool,
    concurrent_load: bool,
) -> EnvironmentMetadata:
    """Capture stable host facts alongside explicitly observed controls."""
    return EnvironmentMetadata(
        captured_at=datetime.now(timezone.utc).isoformat(),
        machine=socket.gethostname(),
        operating_system=platform.system(),
        kernel=platform.release(),
        architecture=platform.machine() or sysconfig.get_platform(),
        python=platform.python_version(),
        commit=commit,
        cpu_governor=cpu_governor,
        gpu_clock=gpu_clock,
        cache_regime=cache_regime,
        benchmark_mode=benchmark_mode,
        concurrent_load=concurrent_load,
    )
