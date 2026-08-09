"""Exact byte accounting for the process transport split."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ByteSplit:
    """Observed byte rates separated by their contract attribution."""

    duration_seconds: float
    samples: int
    batches: int
    logical_sample_bytes: int
    serialized_sample_bytes: int
    batch_bytes: int
    model_input_gbps: float
    irreducible_host_gbps: float
    explicit_overhead_gbps: float
    explicit_total_host_gbps: float


def process_transport_split(
    *,
    duration_seconds: float,
    samples: int,
    batches: int,
    logical_sample_bytes: int,
    serialized_sample_bytes: int,
    batch_bytes: int,
) -> ByteSplit:
    """Account exact explicit copies in the current process delivery path.

    One logical sample write into the arena and one batch materialization are
    the pinned plan's irreducible host writes. The arena payload expansion and
    the two full payload copies on owner delivery are explicit overhead. Python
    serialization internals are intentionally excluded because their physical
    copy count is not instrumented.
    """
    values = (
        duration_seconds,
        samples,
        batches,
        logical_sample_bytes,
        serialized_sample_bytes,
        batch_bytes,
    )
    if any(value <= 0 for value in values):
        raise ValueError("byte split inputs must be positive")
    if serialized_sample_bytes < logical_sample_bytes:
        raise ValueError("serialized samples cannot be smaller than logical samples")
    if batches * batch_bytes != samples * logical_sample_bytes:
        raise ValueError("delivered sample and batch bytes must balance")

    logical_bytes = samples * logical_sample_bytes
    materialized_batch_bytes = batches * batch_bytes
    serialized_bytes = samples * serialized_sample_bytes
    irreducible_host = logical_bytes + materialized_batch_bytes
    explicit_overhead = 3 * serialized_bytes - logical_bytes
    explicit_total = irreducible_host + explicit_overhead
    divisor = duration_seconds * 1_000_000_000.0
    return ByteSplit(
        duration_seconds=duration_seconds,
        samples=samples,
        batches=batches,
        logical_sample_bytes=logical_sample_bytes,
        serialized_sample_bytes=serialized_sample_bytes,
        batch_bytes=batch_bytes,
        model_input_gbps=materialized_batch_bytes / divisor,
        irreducible_host_gbps=irreducible_host / divisor,
        explicit_overhead_gbps=explicit_overhead / divisor,
        explicit_total_host_gbps=explicit_total / divisor,
    )
