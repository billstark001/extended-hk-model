"""Persistent wall budget across attempts; paused time is excluded after clean exit."""

import json
import time
from contextlib import contextmanager

from common.io.json import atomic_json


@contextmanager
def wall_budget(path, seconds, *, clock=time.monotonic, wall=time.time):
    """Yield an absolute monotonic deadline and charge every attempt on exit.

    A killed process cannot finalize its ledger. Such an outstanding attempt is
    conservatively charged through recovery rather than silently granting a new
    budget. The caller must hold the owning run's exclusive session lock.
    """
    saved = json.loads(path.read_text()) if path.exists() else {"seconds": seconds, "used": 0, "active": False}
    if saved["seconds"] != seconds:
        raise ValueError("frozen wall budget changed")
    used = saved["used"]
    if saved["active"]:
        used += max(0, wall() - saved["started_wall"])
    remaining = max(0, seconds - used)
    started = clock()
    atomic_json(path, {"seconds": seconds, "used": used, "active": True, "started_wall": wall()})
    try:
        yield started + remaining
    finally:
        atomic_json(path, {"seconds": seconds, "used": used + max(0, clock() - started), "active": False})
