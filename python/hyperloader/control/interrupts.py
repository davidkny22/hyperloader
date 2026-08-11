"""Unprivileged accelerator-interrupt route discovery."""

from __future__ import annotations

from pathlib import Path

_INTERRUPT_TABLE = Path("/proc/interrupts")
_ACCELERATOR_LABELS = ("nvidia", "amdgpu", "i915", " xe ")


class AcceleratorInterruptRoute:
    """Track cores receiving accelerator interrupts between cadence samples."""

    def __init__(self, path: Path, previous: tuple[int, ...]) -> None:
        self._path = path
        self._previous = previous
        self._cores: tuple[int, ...] = ()

    @classmethod
    def discover(
        cls, path: Path = _INTERRUPT_TABLE
    ) -> AcceleratorInterruptRoute | None:
        """Read the plan-time interrupt baseline without requiring privileges."""
        try:
            previous = _read_accelerator_counts(path)
        except (OSError, ValueError):
            return None
        return cls(path, previous)

    def refresh(self) -> tuple[int, ...]:
        """Return cores with positive interrupt deltas, retaining the last live route."""
        try:
            current = _read_accelerator_counts(self._path)
        except (OSError, ValueError):
            return self._cores
        if len(current) != len(self._previous):
            self._previous = current
            return self._cores
        active = tuple(
            cpu
            for cpu, (before, after) in enumerate(zip(self._previous, current, strict=True))
            if after > before
        )
        self._previous = current
        if active:
            self._cores = active
        return self._cores


def _read_accelerator_counts(path: Path) -> tuple[int, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("interrupt table is empty")
    cpus = tuple(token for token in lines[0].split() if token.startswith("CPU"))
    if not cpus:
        raise ValueError("interrupt table has no CPU header")
    totals = [0] * len(cpus)
    matched = False
    for line in lines[1:]:
        lowered = f" {line.lower()} "
        if not any(label in lowered for label in _ACCELERATOR_LABELS):
            continue
        fields = line.split()
        if len(fields) < len(cpus) + 1 or not fields[0].endswith(":"):
            continue
        try:
            counts = tuple(int(value) for value in fields[1 : len(cpus) + 1])
        except ValueError:
            continue
        matched = True
        for cpu, count in enumerate(counts):
            totals[cpu] += count
    if not matched:
        raise ValueError("interrupt table has no accelerator rows")
    return tuple(totals)
