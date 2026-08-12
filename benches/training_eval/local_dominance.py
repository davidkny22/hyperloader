"""Steady-machine controls for the local dominance card."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benches.benchmark_protocol import EnvironmentMetadata, capture_environment
from benches.benchmark_protocol.matrix import workload_names

from .ambient import AmbientProbe, compare_ambient
from .decision import TrainingDecision
from .lease import FileLease
from .output import write_result

LOCAL_REFERENCES = ("torch", "spdl")


def run_local_dominance(
    *,
    output: Path,
    lock_path: Path,
    prior_ambient: AmbientProbe,
    current_ambient: AmbientProbe,
    null_decision: TrainingDecision,
    commit: str,
    lease_claimant: str,
    cpu_governor: str,
    gpu_clock: str,
    worker_cpus: tuple[int, ...],
    torchvision_version: str,
    smoke: bool,
    campaign: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Run the public six-cell matrix only after ambient and lease acceptance."""
    if lock_path.name != "LOCAL-LOCK":
        raise ValueError("local dominance requires the exact LOCAL-LOCK path")
    if output.suffix:
        raise ValueError("local dominance output must be a result directory")
    if null_decision.status != "pass" or null_decision.mode != "absolute":
        raise RuntimeError("local dominance requires a passing absolute null decision")
    null_band_percent = max(
        abs(null_decision.lower_percent), abs(null_decision.upper_percent)
    )
    ambient = compare_ambient(
        prior_ambient,
        current_ambient,
        null_band_percent=null_band_percent,
    )
    if ambient.status != "pass":
        raise RuntimeError("local ambient probe is outside the accepted null band")
    with FileLease.claim(
        lock_path,
        claimant=lease_claimant,
        purpose="local training dominance measurement",
    ) as lease:
        environment: EnvironmentMetadata = capture_environment(
            commit=commit,
            cpu_governor=cpu_governor,
            gpu_clock=gpu_clock,
            cache_regime="warm",
            benchmark_mode=True,
            concurrent_load=False,
        )
        summary = campaign(
            output=output,
            environment=environment,
            worker_cpus=worker_cpus,
            torchvision_version=torchvision_version,
            workloads=workload_names(),
            references=LOCAL_REFERENCES,
            smoke=smoke,
            capture_cpuidle=False,
            smoke_ordinal=0,
        )
        lease.verify()
        write_result(
            output / "local-session.json",
            {
                "kind": "local-dominance-session",
                "lease": {
                    "timestamp": lease.record.timestamp.isoformat(),
                    "claimant": lease.record.claimant,
                    "token": lease.record.token,
                    "purpose": lease.record.purpose,
                },
                "ambient": {
                    "prior": asdict(prior_ambient),
                    "current": asdict(current_ambient),
                    "decision": asdict(ambient),
                },
                "null_decision": asdict(null_decision),
                "summary": summary,
            },
        )
        return summary
