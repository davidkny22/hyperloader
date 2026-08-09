"""Record and stop owner-authorized Spark GUI processes before measurement."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

GUI_MARKERS = (
    "/usr/lib/xorg/Xorg",
    "/usr/bin/gnome-shell",
    "gnome-shell-calendar-server",
    "org.gnome.Shell.Notifications",
    "org.gnome.ScreenSaver",
    "ding@rastersoft.com",
    "gnome-terminal-server",
    "gnome-remote-desktop-daemon",
)


@dataclass(frozen=True)
class ProcessRecord:
    """One resolved process selected by an owner-authorized marker."""

    pid: int
    user: str
    command: str
    arguments: str


def process_inventory() -> list[ProcessRecord]:
    """Read the current process table without changing machine state."""
    completed = subprocess.run(
        ["ps", "-eo", "pid=,user=,comm=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    records = []
    for line in completed.stdout.splitlines():
        pid, user, command, arguments = line.split(maxsplit=3)
        records.append(ProcessRecord(int(pid), user, command, arguments))
    return records


def gui_processes(records: list[ProcessRecord]) -> list[ProcessRecord]:
    """Select only the explicitly authorized GUI process family."""
    return [
        record
        for record in records
        if any(marker in record.arguments for marker in GUI_MARKERS)
        and "spark_machine_quiet.py" not in record.arguments
    ]


def quiet_machine(evidence: Path) -> None:
    """Terminate selected GUI processes, escalate survivors, and record them."""
    selected = gui_processes(process_inventory())
    if selected:
        subprocess.run(
            ["sudo", "-n", "kill", "-TERM", *(str(item.pid) for item in selected)],
            check=True,
        )
        time.sleep(2.0)
        live_pids = {record.pid for record in process_inventory()}
        survivors = [record for record in selected if record.pid in live_pids]
        if survivors:
            subprocess.run(
                [
                    "sudo",
                    "-n",
                    "kill",
                    "-KILL",
                    *(str(item.pid) for item in survivors),
                ],
                check=True,
            )
            time.sleep(1.0)
    remaining = gui_processes(process_inventory())
    document = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "killed": [asdict(record) for record in selected],
        "remaining": [asdict(record) for record in remaining],
    }
    evidence.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if remaining:
        raise RuntimeError("owner-authorized GUI processes remain after quieting")


def main() -> None:
    """Resolve owner-authorized GUI targets and preserve the kill record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    quiet_machine(arguments.evidence)


if __name__ == "__main__":
    main()
