"""Bounded pure-Python execution-cost profiles."""

from __future__ import annotations

import json
import math
from pathlib import Path


class CostProfile:
    """Retain exact or bucketed EMA costs under a byte ceiling."""

    def __init__(self, position_count: int, max_bytes: int, alpha: float) -> None:
        if position_count < 0 or max_bytes < 0:
            raise ValueError("profile sizes must be nonnegative")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("profile alpha must be in (0, 1]")
        self._position_count = position_count
        self._alpha = alpha
        self._bucket_count = min(position_count, max_bytes // 8)
        self._degraded = self._bucket_count < position_count
        self._values: dict[int, float] = {}

    @classmethod
    def load(
        cls, path: Path, position_count: int, max_bytes: int, alpha: float
    ) -> CostProfile:
        """Load one compatible profile or raise on identity drift."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        profile = cls(position_count, max_bytes, alpha)
        if (
            payload.get("position_count") != position_count
            or payload.get("bucket_count") != profile._bucket_count
            or not math.isclose(float(payload.get("alpha")), alpha)
        ):
            raise ValueError("cost profile configuration changed")
        profile._values = {
            int(key): float(value) for key, value in payload["values"].items()
        }
        return profile

    def observe(self, position: int, cost_ns: int) -> None:
        """Update one exact or bucketed EMA observation."""
        bucket = self._bucket(position)
        value = float(cost_ns)
        previous = self._values.get(bucket)
        self._values[bucket] = (
            value
            if previous is None
            else self._alpha * value + (1 - self._alpha) * previous
        )

    def estimate(self, position: int) -> float | None:
        """Return the current estimate for one position."""
        return self._values.get(self._bucket(position))

    def statistics(self) -> tuple[float, float, int] | None:
        """Return mean, conservative p99.9, and populated bucket count."""
        if not self._values:
            return None
        values = sorted(self._values.values())
        index = min(len(values) - 1, math.ceil(0.999 * len(values)) - 1)
        return sum(values) / len(values), values[index], len(values)

    def save(self, path: Path) -> None:
        """Persist the profile through an atomic sibling replacement."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "alpha": self._alpha,
                    "bucket_count": self._bucket_count,
                    "position_count": self._position_count,
                    "values": self._values,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    @property
    def degraded(self) -> bool:
        """Return whether multiple positions share cost buckets."""
        return self._degraded

    @property
    def payload_bytes(self) -> int:
        """Return the bounded in-memory estimate payload."""
        return self._bucket_count * 8

    def _bucket(self, position: int) -> int:
        if not 0 <= position < self._position_count:
            raise ValueError("profile position is outside its domain")
        if self._bucket_count == 0:
            raise ValueError("profile byte budget admits no observations")
        return position % self._bucket_count
