"""Thread-safe task timing and throughput estimates without model dependencies."""

import sys
import threading
import time


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
