"""Passive observation of a stock Torch DataLoader."""

from __future__ import annotations

from typing import Any

from .workers import snapshot_workers


def observe_stock(loader: Any) -> dict[str, object]:
    """Read stock loader and iterator state without advancing either object."""
    iterator = getattr(loader, "_iterator", None)
    workers = () if iterator is None else tuple(getattr(iterator, "_workers", ()))
    capacity = _capacity(loader, iterator)
    ready = _ready_batches(iterator)
    outstanding = _integer(iterator, "_tasks_outstanding")
    in_flight = None if outstanding is None else max(0, outstanding - ready)
    occupancy = None if capacity == 0 else ready / capacity
    saturation = {
        "basis": "ready results divided by the configured prefetch capacity",
        "capacity_batches": capacity,
        "in_flight_batches": in_flight,
        "margin_batches": None if capacity == 0 else ready - 1,
        "occupancy_fraction": occupancy,
        "ready_batches": ready,
    }
    blocking = {
        "basis": (
            "Stock DataLoader exposes current queue state but no cumulative consumer "
            "wait duration to a passive observer."
        ),
        "fraction": None,
        "currently_blocked": (
            iterator is not None and bool(outstanding) and ready == 0
        ),
        "wait_ns": None,
    }
    gil_release = {
        "basis": "A passive stock loader snapshot does not sample interpreter ownership.",
        "fraction": None,
    }
    return {
        "loader_kind": "torch",
        "blocking": blocking,
        "ceiling_binds": [],
        "gil_release": gil_release,
        "saturation": saturation,
        "steps": _steps(loader, ready, in_flight),
        "workers": snapshot_workers(workers),
    }


def _capacity(loader: Any, iterator: Any | None) -> int:
    if iterator is None:
        return 0
    workers = int(getattr(loader, "num_workers", 0))
    prefetch = getattr(loader, "prefetch_factor", None) or 2
    return workers * int(prefetch)


def _ready_batches(iterator: Any | None) -> int:
    if iterator is None:
        return 0
    task_info = getattr(iterator, "_task_info", {})
    return sum(len(value) == 2 for value in task_info.values())


def _integer(value: Any | None, attribute: str) -> int | None:
    raw = None if value is None else getattr(value, attribute, None)
    return raw if isinstance(raw, int) else None


def _steps(loader: Any, ready: int, in_flight: int | None) -> list[dict[str, object]]:
    return [
        {"name": "source", "detail": type(loader.dataset).__qualname__},
        {"name": "sampler", "detail": type(loader.sampler).__qualname__},
        {
            "name": "worker_fetch",
            "detail": {
                "in_flight_batches": in_flight,
                "num_workers": loader.num_workers,
            },
        },
        {"name": "pin_memory", "detail": bool(loader.pin_memory)},
        {"name": "delivery", "detail": {"ready_batches": ready}},
    ]
