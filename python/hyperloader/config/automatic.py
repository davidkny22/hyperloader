"""Values derived from plan-time measurements."""

from __future__ import annotations

from typing import Final


class _Auto:
    """Represent a value derived from measured quantities at plan time."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "auto"


AUTO: Final = _Auto()
"""The singleton used for values derived at plan time."""

Auto = _Auto
AutoInt = int | _Auto
AutoFloat = float | _Auto
AutoBytes = int | _Auto


def _require_nonnegative_int(name: str, value: AutoInt) -> None:
    if value is not AUTO and (isinstance(value, bool) or value < 0):
        raise ValueError(f"{name} must be auto or a nonnegative integer")
