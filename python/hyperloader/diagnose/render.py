"""Human-readable diagnosis rendering."""

from __future__ import annotations


def render_diagnosis(record: dict[str, object]) -> str:
    """Render the stable machine record without adding new claims."""
    saturation = record["saturation"]
    blocking = record["blocking"]
    promotion = record["promotion"]
    assert isinstance(saturation, dict)
    assert isinstance(blocking, dict)
    assert isinstance(promotion, dict)
    expected = promotion["expected_gain"]
    assert isinstance(expected, dict)
    lines = [
        "Loader diagnosis",
        f"Loader kind: {record['loader_kind']}.",
        f"Observation mode: {record['observation_mode']}.",
        f"Ready-capacity fraction: {_format(saturation.get('occupancy_fraction'))}.",
        f"Realized blocking fraction: {_format(blocking.get('fraction'))}.",
        f"Observed workers: {len(record['workers'])}.",
        f"Causal ceiling binds: {record['ceiling_binds'] or 'none'}.",
        f"Attributed cause: {record['attribution']['cause']}.",
        "Expected gain: unreported. " + str(expected["basis"]),
    ]
    probe = record.get("probe")
    if isinstance(probe, dict):
        lines.append(
            "Active probe: "
            f"{probe['consumed_batches']} batches consumed in {probe['elapsed_ns']} ns."
        )
    return "\n".join(lines)


def _format(value: object) -> str:
    return "not measured" if value is None else f"{float(value):.6f}"
