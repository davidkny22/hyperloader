"""Token-checked local machine lease with Spark-lock semantics."""

from __future__ import annotations

import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Self

_RECORD = re.compile(
    r"^claimed (?P<timestamp>\S+) \| (?P<claimant>[^|]+) \| token "
    r"(?P<token>[0-9a-f]{8}) \| (?P<purpose>.+)$"
)


class LeaseUnavailable(RuntimeError):
    """The machine is owned by another live or non-stale claim."""


@dataclass(frozen=True)
class LeaseRecord:
    """One parsed lease line."""

    timestamp: datetime
    claimant: str
    token: str
    purpose: str

    @classmethod
    def parse(cls, text: str) -> LeaseRecord:
        """Parse the exact shared lease grammar."""
        match = _RECORD.fullmatch(text.strip())
        if match is None:
            raise ValueError("invalid machine lease record")
        timestamp = datetime.fromisoformat(match.group("timestamp"))
        if timestamp.tzinfo is None:
            raise ValueError("machine lease timestamp must include a timezone")
        return cls(
            timestamp=timestamp,
            claimant=match.group("claimant").strip(),
            token=match.group("token"),
            purpose=match.group("purpose").strip(),
        )

    def render(self) -> str:
        """Render the shared one-line lease grammar."""
        return (
            f"claimed {self.timestamp.isoformat()} | {self.claimant} | "
            f"token {self.token} | {self.purpose}\n"
        )


class FileLease:
    """Claim, refresh, verify, and release a token-owned machine lock."""

    def __init__(self, path: Path, record: LeaseRecord) -> None:
        self.path = path
        self.record = record

    @classmethod
    def claim(
        cls,
        path: Path,
        *,
        claimant: str,
        purpose: str,
        active_process: Callable[[], bool] = lambda: True,
        now: datetime | None = None,
        verify_delay_seconds: float = 0.25,
    ) -> FileLease:
        """Claim an absent lock or supersede a proven stale inactive claim."""
        instant = now or datetime.now(timezone.utc)
        token = secrets.token_hex(4)
        record = LeaseRecord(instant, claimant, token, purpose)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(record.render())
        except FileExistsError:
            existing = LeaseRecord.parse(path.read_text(encoding="utf-8"))
            age_seconds = (instant - existing.timestamp).total_seconds()
            if age_seconds <= 3600 or active_process():
                raise LeaseUnavailable(
                    f"machine lease belongs to token {existing.token}"
                ) from None
            path.write_text(record.render(), encoding="utf-8")
        if verify_delay_seconds > 0:
            time.sleep(verify_delay_seconds)
        lease = cls(path, record)
        lease.verify()
        return lease

    def verify(self) -> None:
        """Require the on-disk token to remain this lease's token."""
        observed = LeaseRecord.parse(self.path.read_text(encoding="utf-8"))
        if observed.token != self.record.token:
            raise LeaseUnavailable("machine lease token lost the claim race")

    def refresh(self, *, now: datetime | None = None) -> None:
        """Refresh a still-owned claim without changing its token."""
        self.verify()
        self.record = LeaseRecord(
            now or datetime.now(timezone.utc),
            self.record.claimant,
            self.record.token,
            self.record.purpose,
        )
        self.path.write_text(self.record.render(), encoding="utf-8")
        self.verify()

    def release(self) -> None:
        """Delete only the lock still owned by this token."""
        self.verify()
        self.path.unlink()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
