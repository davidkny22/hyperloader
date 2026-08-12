"""Evidence-bounded execution promotion advice."""

from __future__ import annotations

from typing import Any


def promotion_record(
    loader: Any,
    *,
    loader_kind: str,
    saturation: dict[str, object],
    gil_release: dict[str, object],
) -> dict[str, object]:
    """Describe a candidate configuration without inventing a speed claim."""
    if loader_kind == "torch":
        candidate = {
            "mode": "native",
            "num_workers": "auto",
            "pin_memory": bool(getattr(loader, "pin_memory", False)),
            "thread_safe": False,
        }
        purpose = "promote the observed stock configuration to native execution"
    else:
        candidate = {
            "mode": getattr(loader, "mode", "native"),
            "num_workers": _worker_setting(getattr(loader, "num_workers", "auto")),
            "pin_memory": bool(getattr(loader, "pin_memory", False)),
            "thread_safe": bool(getattr(loader, "_sample_thread_safe", False)),
        }
        purpose = "retain the observed hyperloader execution contract"
    return {
        "candidate_config": candidate,
        "purpose": purpose,
        "evidence": [
            {
                "signal": "queue_saturation",
                "value": saturation.get("occupancy_fraction"),
                "basis": saturation.get("basis"),
            },
            {
                "signal": "gil_release_fraction",
                "value": gil_release.get("fraction"),
                "basis": gil_release.get("basis"),
            },
        ],
        "expected_gain": {
            "lower_percent": None,
            "upper_percent": None,
            "basis": (
                "No matched decision-protocol baseline was observed, so this report "
                "does not publish a numerical gain estimate."
            ),
        },
    }


def _worker_setting(value: object) -> int | str:
    return value if isinstance(value, int) else "auto"
