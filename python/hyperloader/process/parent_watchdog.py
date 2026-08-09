"""Independent worker supervision for abrupt owner-process death."""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time

WATCHDOG_SECONDS = 0.05


def start_parent_watchdog() -> None:
    """End the worker promptly if its multiprocessing parent disappears."""
    parent = mp.parent_process()
    if parent is None:
        return

    def monitor() -> None:
        while parent.is_alive():
            time.sleep(WATCHDOG_SECONDS)
        os._exit(0)

    threading.Thread(
        target=monitor,
        name="hyperloader-parent-watchdog",
        daemon=True,
    ).start()
