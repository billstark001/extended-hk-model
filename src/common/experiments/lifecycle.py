"""Exclusive sessions, stage dispatch and durable attempt status for all studies."""

import fcntl
import json
import os
import signal
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from common.io.json import atomic_json


def read_owned(path: Path, root: Path) -> dict[str, Any]:
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"run metadata links outside its directory: {path}")
    return cast(dict[str, Any], json.loads(path.read_text()))


@contextmanager
def session_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "session.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"run already active: {directory}") from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def recover_task_lock(directory: Path) -> None:
    path = directory / "active.lock"
    if not path.exists():
        return
    pid = int(path.read_text())
    if pid <= 0:
        raise ValueError(f"invalid PID lock: {path}")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        path.unlink()
    else:
        raise RuntimeError(f"run already active (PID {pid}): {directory}")


@contextmanager
def task_lock(directory: Path, store=None):
    from common.runtime.process import acquire_lock, release_lock

    path = directory / "active.lock"

    def operation():
        return acquire_lock(path)

    store.read(operation) if store else operation()
    try:
        yield
    finally:

        def operation():
            return release_lock(path)

        try:
            store.read(operation) if store else operation()
        except OSError:
            # The owning PID is recoverable if an external volume disappears.
            pass


@contextmanager
def interruptible():
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class Workflow:
    """One stage dispatcher; callbacks supply study-specific computation and I/O."""

    def __init__(self, stages: Mapping[str, Callable[[], Any]], status_path: Path | None = None, **metadata):
        self.stages = dict(stages)
        self.guard = lambda operation: operation()
        self.status_path = status_path
        self.status = {**metadata, "state": "running", "action": "verify"}

    def _write(self):
        if self.status_path is not None:
            atomic_json(self.status_path, self.status)

    def dispatch(self, action: str):
        if action not in self.stages:
            raise ValueError(f"unavailable workflow stage: {action}")
        self.status.update(action=action, state="running")
        self._write()
        try:
            result = self.guard(self.stages[action])
        except BaseException as error:
            self.status.update(
                state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error)
            )
            self._write()
            raise
        self.status.pop("error", None)
        self.status["state"] = "partial" if isinstance(result, dict) and result.get("missing") else "complete"
        self._write()
        return result

    def sequence(self, actions, *, after_failure=None):
        for action in actions:
            try:
                self.dispatch(action)
            except Exception as error:
                if after_failure is not None:
                    after_failure(action, error)
                raise


class TaskJournal:
    """Append transitions locally and publish a compact terminal snapshot."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def event(self, identifier: str, state: str, **details):
        entry = {"id": identifier, "state": state, "time": time.time(), **details}
        with self.lock, (self.directory / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            self.records[identifier] = entry

    def finish(self, tasks, state="complete"):
        from .plans import coverage

        with self.lock:
            records = list(self.records.values())
        successful = [row["id"] for row in records if row["state"] in ("complete", "cached")]
        result = {**coverage(tasks, successful), "state": state, "records": records}
        if state == "complete" and result["missing"]:
            result["state"] = "partial"
        atomic_json(self.directory / "status.json", result)
        return result


class StageJournals:
    """Route a dependency stream to existing per-stage journals and coverage files."""

    def __init__(self, directory, stages, *, prefix=""):
        self.directory, self.stages, self.prefix = Path(directory), stages, prefix
        self.journals = {s["name"]: TaskJournal(self.directory / (prefix + s["name"])) for s in stages}
        self.owners = {identifier: s["name"] for s in stages for identifier in s["task_ids"]}

    def event(self, identifier, state, **details):
        self.journals[self.owners[identifier]].event(identifier, state, **details)

    def publish(self, completed):
        from .plans import coverage

        for stage in self.stages:
            identifiers = stage["task_ids"]
            counts = (
                coverage(({"id": i} for i in identifiers), set(identifiers) & completed)
                if identifiers
                else dict(planned=0, completed=0, missing=[])
            )
            atomic_json(
                self.directory / (self.prefix + stage["name"]) / "coverage.json",
                {**counts, "state": "partial" if counts["missing"] else "complete"},
            )
