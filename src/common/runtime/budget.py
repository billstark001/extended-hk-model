"""Persistent wall budget across attempts; paused time is excluded after clean exit."""

import json
import math
import os
import time
from contextlib import contextmanager

from common.io.json import atomic_json


def extend_wall_budget(path, *, initial_seconds, seconds=None, metadata=None):
    """Idempotently raise a cumulative limit without changing accumulated usage.

    Callers hold their session lock. A stale active ledger keeps its start time
    for conservative recovery by wall_budget. The audit records the explicit
    request before the updated ledger is published.
    """
    saved = json.loads(path.read_text()) if path.exists() else None
    previous = saved["seconds"] if saved else initial_seconds
    target = previous if seconds is None else seconds
    if not math.isfinite(target) or target <= 0 or target < previous:
        raise ValueError("wall limit must be finite, positive and cannot reduce the cumulative limit")
    if target != previous:
        ledger = saved or {"used": 0, "active": False}
        audit = path.with_name(path.stem + "_extensions.jsonl")
        with audit.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        **(metadata or {}),
                        "time": time.time(),
                        "event": "limit_requested",
                        "previous_seconds": previous,
                        "seconds": target,
                        "used": ledger["used"],
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(path, {**ledger, "seconds": target})
    return target


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
