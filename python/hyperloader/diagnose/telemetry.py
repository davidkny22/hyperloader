"""Non-mutating telemetry composition for passive diagnosis."""

from __future__ import annotations

from typing import Any

from hyperloader.telemetry.runtime import telemetry_snapshot


def passive_telemetry(loader: Any) -> dict[str, object]:
    """Read loader telemetry without flushing its live delivery buffer."""
    snapshot = telemetry_snapshot(
        loader._telemetry,
        loader._last_controller_report,
        None,
    )
    execution_dataset = getattr(loader, "_execution_dataset", None)
    memory_report = getattr(execution_dataset, "memory_report", None)
    memory_ledger = getattr(loader, "_memory_ledger", None)
    if memory_report is not None:
        snapshot["memory"] = memory_report()
    elif memory_ledger is not None:
        snapshot["memory"] = memory_ledger.report()
    pinned = getattr(loader, "_pinned_delivery", None)
    if pinned is not None and pinned.reports_selection:
        memory = snapshot.setdefault("memory", {})
        if isinstance(memory, dict):
            pinned.compose_memory_report(memory)
    current = snapshot.get("current")
    if isinstance(current, dict):
        keeper = getattr(loader, "_machine_keeper", None)
        current["machine_keeping_duty"] = (
            0.0 if keeper is None else float(keeper.duty())
        )
    return snapshot
