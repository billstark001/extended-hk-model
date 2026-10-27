"""Thread-safe task timing and throughput estimates without model dependencies."""

import json
import sys
import threading
import time
from pathlib import Path


def duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


class Progress:
    def __init__(self, total, *, stream=None, clock=time.monotonic):
        self.total, self.stream, self.clock = total, stream or sys.stdout, clock
        self.started = clock()
        self.starts, self.finished = {}, set()
        self.computed = 0
        self.lock = threading.Lock()

    def event(self, identifier, state):
        with self.lock:
            now = self.clock()
            if state == "running":
                self.starts.setdefault(identifier, now)
                return
            if state not in ("complete", "cached", "failed", "deferred") or identifier in self.finished:
                return
            self.finished.add(identifier)
            if state != "cached":
                self.computed += 1
            elapsed = now - self.started
            remaining = self.total - len(self.finished)
            eta = duration(elapsed * remaining / self.computed) if self.computed else "—"
            span = duration(now - self.starts.get(identifier, now))
            mark = {"complete": "✓", "cached": "↪", "failed": "✗", "deferred": "…"}[state]
            count = f"{len(self.finished)}/{self.total}"
            print(
                f"{mark} [{count}] {identifier} · {state} · {span} · elapsed {duration(elapsed)} · ETA {eta}",
                file=self.stream,
                flush=True,
            )


class JournalIndex:
    def __init__(self, path, *, deferred_errors=()):
        self.path, self.offset, self.buffer = Path(path), 0, b""
        self.deferred_errors = deferred_errors
        self.complete, self.active, self.failed, self.deferred = set(), {}, set(), set()
        self.starts, self.durations, self.finished = {}, {}, {}

    def event(self, identifier, state, **details):
        if not identifier:
            return
        stamp = details.get("time", time.time())
        if state == "running":
            self.starts[identifier] = stamp
            if identifier not in self.complete:
                self.active[identifier] = stamp
        if state in ("complete", "cached"):
            self.complete.add(identifier)
            self.failed.discard(identifier)
            self.deferred.discard(identifier)
            if state == "complete":
                self.durations[identifier] = max(0, stamp - self.starts.get(identifier, stamp))
                self.finished[identifier] = stamp
        if state in ("failed", "deferred"):
            budget = state == "deferred" or any(message in details.get("error", "") for message in self.deferred_errors)
            (self.deferred if budget else self.failed).add(identifier)
        if state in ("complete", "cached", "failed", "deferred"):
            self.active.pop(identifier, None)

    def refresh(self):
        if not self.path.exists():
            return
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            data = self.buffer + stream.read()
            self.offset = stream.tell()
        lines = data.split(b"\n")
        self.buffer = lines.pop()
        for line in lines:
            if not line:
                continue
            event = json.loads(line)
            identifier, state = event.pop("task", None), event.pop("state")
            self.event(identifier, state, **event)


def remaining_work(tasks, completed, active, predict, workers, *, now=None, overhead=1.15):
    """Aggregate serial cost predictions without pretending overruns are zero work."""
    if workers < 1:
        raise ValueError("positive workers required")
    now = time.time() if now is None else now
    serial, unknown, overruns = 0.0, 0, 0
    for identifier in tasks.keys() - completed:
        prediction = predict(tasks[identifier])
        if prediction is None:
            unknown += 1
        elif identifier in active and now - active[identifier] >= prediction:
            unknown += 1
            overruns += 1
        else:
            serial += max(0.01, prediction - max(0, now - active[identifier])) if identifier in active else prediction
    known = serial / workers * overhead
    return dict(
        estimated_remaining_seconds=known if not unknown else None,
        known_remaining_seconds=known,
        unknown_cost_tasks=unknown,
        active_estimate_overruns=overruns,
    )
