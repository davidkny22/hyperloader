"""Run the six-cell training dominance card on the local GPU host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_protocol.matrix import workload_names
from dominance_campaign import run_campaign
from training_eval.ambient import AmbientProbe
from training_eval.decision import TrainingDecision
from training_eval.local_dominance import run_local_dominance


def main() -> None:
    """Validate steady-machine controls and execute the proven matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--lease-claimant", required=True)
    parser.add_argument("--cpu-governor", required=True)
    parser.add_argument("--gpu-clock", required=True)
    parser.add_argument("--worker-cpus", type=int, nargs="+", required=True)
    parser.add_argument("--torchvision-version", required=True)
    parser.add_argument("--prior-ambient", type=Path, required=True)
    parser.add_argument("--current-ambient", type=Path, required=True)
    parser.add_argument("--null-decision", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("LOCAL-LOCK"))
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    prior = AmbientProbe(**json.loads(arguments.prior_ambient.read_text()))
    current = AmbientProbe(**json.loads(arguments.current_ambient.read_text()))
    null_decision = TrainingDecision(**json.loads(arguments.null_decision.read_text()))
    summary = run_local_dominance(
        output=arguments.output,
        lock_path=arguments.lock,
        prior_ambient=prior,
        current_ambient=current,
        null_decision=null_decision,
        commit=arguments.commit,
        lease_claimant=arguments.lease_claimant,
        cpu_governor=arguments.cpu_governor,
        gpu_clock=arguments.gpu_clock,
        worker_cpus=tuple(arguments.worker_cpus),
        torchvision_version=arguments.torchvision_version,
        smoke=arguments.smoke,
        campaign=run_campaign,
    )
    print(
        json.dumps({"workloads": workload_names(), "summary": summary}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
