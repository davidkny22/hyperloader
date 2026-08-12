"""Passive observation of a hyperloader DataLoader."""

from __future__ import annotations

from typing import Any

from .telemetry import passive_telemetry
from .workers import snapshot_workers


def observe_native(loader: Any) -> dict[str, object]:
    """Compose public telemetry and live scheduler state without advancing delivery."""
    telemetry = passive_telemetry(loader)
    summary = telemetry.get("current") or telemetry.get("last_epoch") or {}
    frontier = _frontier_report(loader)
    capacity = _number(frontier.get("final_depth"))
    occupied = _live_occupied(loader, frontier)
    saturation = {
        "basis": "live admitted frontier occupancy divided by its active depth",
        "capacity_batches": capacity,
        "in_flight_batches": occupied,
        "margin_batches": None if occupied is None else occupied - 1,
        "occupancy_fraction": (
            None if not capacity or occupied is None else occupied / capacity
        ),
        "ready_batches": None,
    }
    blocking = {
        "basis": "native scheduler wait time divided by active loader time",
        "fraction": _number(frontier.get("stall_fraction")),
        "wait_ns": _integer(frontier.get("wait_ns")),
    }
    controller = telemetry.get("controller")
    binding = controller.get("binding") if isinstance(controller, dict) else None
    gil_events = (
        summary.get("gil_restore_events") if isinstance(summary, dict) else None
    )
    gil_release = {
        "basis": (
            "Native telemetry records GIL restoration events but does not infer a time "
            "fraction without an active interpreter probe."
        ),
        "fraction": None,
        "restore_events": gil_events,
    }
    return {
        "loader_kind": "hyperloader",
        "blocking": blocking,
        "ceiling_binds": [] if binding is None else [binding],
        "gil_release": gil_release,
        "saturation": saturation,
        "steps": _steps(loader, summary, frontier),
        "telemetry": telemetry,
        "workers": snapshot_workers(_worker_processes(loader)),
    }


def _frontier_report(loader: Any) -> dict[str, object]:
    schedule = _active_schedule(loader)
    if schedule is not None:
        return dict(schedule.report())
    report = getattr(loader, "_last_frontier_report", None)
    return {} if report is None else dict(report)


def _live_occupied(loader: Any, frontier: dict[str, object]) -> int | None:
    schedule = _active_schedule(loader)
    if schedule is not None:
        return int(schedule.occupied)
    return _integer(frontier.get("max_occupied"))


def _active_schedule(loader: Any) -> Any | None:
    reference = getattr(loader, "_active_iterator_ref", None)
    iterator = None if reference is None else reference()
    seen: set[int] = set()
    while iterator is not None and id(iterator) not in seen:
        seen.add(id(iterator))
        schedule = getattr(iterator, "_schedule", None)
        if schedule is not None:
            return schedule
        nested = getattr(iterator, "_iterator", None)
        if nested is iterator:
            break
        iterator = nested
    return None


def _worker_processes(loader: Any) -> tuple[Any, ...]:
    for attribute in ("_process_pool", "_compat_lane_pool"):
        pool = getattr(loader, attribute, None)
        if pool is not None:
            worker_set = getattr(pool, "_worker_set", None)
            processes = getattr(worker_set, "processes", None)
            if processes is not None:
                return tuple(processes)
            return tuple(pool.worker_pids)
    return ()


def _steps(
    loader: Any, summary: dict[str, object], frontier: dict[str, object]
) -> list[dict[str, object]]:
    plan = getattr(loader, "_plan", None)
    return [
        {"name": "source", "detail": type(loader.dataset).__qualname__},
        {"name": "plan", "detail": None if plan is None else type(plan).__qualname__},
        {"name": "frontier", "detail": frontier},
        {
            "name": "delivery",
            "detail": {
                "batches": summary.get("delivered_batches"),
                "bytes": summary.get("delivered_bytes"),
                "rate": summary.get("delivery_rate"),
            },
        },
    ]


def _number(value: object) -> float | int | None:
    return value if isinstance(value, (float, int)) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) else None
