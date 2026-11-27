"""Exclusive sessions, stage dispatch and durable attempt status for all studies."""

import json
import os
import signal
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from common.io.json import atomic_json
from common.io.json import json_default
from common.runtime.failures import ProtocolError


@dataclass(kw_only=True, slots=True)
class WorkflowStatus:
    state: str = "running"
    action: str = "verify"
    profile: str | None = None
    workers: int | None = None
    resume: bool | None = None
    operation: str | None = None
    error: str | None = None

    def __wire__(self, *, preserve_arrays=False):
        return {item.name: getattr(self, item.name) for item in fields(self)
                if getattr(self, item.name) is not None}


@dataclass(frozen=True, kw_only=True, slots=True)
class RunRecord:
    id: str
    state: str
    time: float
    error: str | None = None
    reason: str | None = None
    attempt: int | None = None

    def __wire__(self, *, preserve_arrays=False):
        return {item.name: getattr(self, item.name) for item in fields(self)
                if getattr(self, item.name) is not None}


@dataclass(kw_only=True, slots=True)
class RunStatus:
    planned: int
    completed: int
    missing: tuple[str, ...]
    state: str
    records: tuple[RunRecord, ...]

    def __wire__(self, *, preserve_arrays=False):
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["missing"] = list(self.missing)
        result["records"] = [item.__wire__(preserve_arrays=preserve_arrays) for item in self.records]
        return result


def read_owned(path: Path, root: Path) -> dict[str, Any]:
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"run metadata links outside its directory: {path}")
    return cast(dict[str, Any], json.loads(path.read_text()))


@contextmanager
def session_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "session.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
        else:
            import fcntl

        try:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"run already active: {directory}") from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()).encode("ascii"))
            handle.flush()
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def recover_task_lock(directory: Path) -> None:
    from common.runtime.process import process_exists

    path = directory / "active.lock"
    if not path.exists():
        return
    pid = int(path.read_text())
    if pid <= 0:
        raise ValueError(f"invalid PID lock: {path}")
    if process_exists(pid):
        raise RuntimeError(f"run already active (PID {pid}): {directory}")
    path.unlink()


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

    def __init__(self, stages: Mapping[str, Callable[[], object]], status_path: Path | None = None,
                 *, profile: str | None = None, workers: int | None = None,
                 resume: bool | None = None, operation: str | None = None,
                 result_state: Callable[[object], str] | None = None):
        self.stages = dict(stages)
        self.guard = lambda operation: operation()
        self.status_path = status_path
        self.status = WorkflowStatus(profile=profile, workers=workers, resume=resume, operation=operation)
        self.result_state = result_state

    def _write(self):
        if self.status_path is not None:
            atomic_json(self.status_path, self.status)

    def dispatch(self, action: str):
        if action not in self.stages:
            raise ValueError(f"unavailable workflow stage: {action}")
        self.status.action, self.status.state = action, "running"
        self._write()
        try:
            result = self.guard(self.stages[action])
            state = (self.result_state(result) if self.result_state is not None else
                     result.state if isinstance(result, RunStatus) else "complete")
        except BaseException as error:
            self.status.state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            self.status.error = str(error)
            self._write()
            raise
        self.status.error = None
        self.status.state = state
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
        self.records: dict[str, RunRecord] = {}
        self.lock = threading.Lock()

    def event(self, identifier: str, state: str, *, error: str | None = None,
              reason: str | None = None, attempt: int | None = None):
        entry = RunRecord(id=identifier, state=state, time=time.time(), error=error, reason=reason, attempt=attempt)
        with self.lock, (self.directory / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(entry, sort_keys=True, default=json_default) + "\n")
            handle.flush()
            self.records[identifier] = entry

    def finish(self, task_ids: Iterable[str], state="complete") -> RunStatus:
        expected = tuple(task_ids)
        if any(type(identifier) is not str for identifier in expected) or len(set(expected)) != len(expected):
            raise ProtocolError("journal task IDs must be distinct strings")
        with self.lock:
            records = tuple(self.records.values())
        successful = {row.id for row in records if row.state in ("complete", "cached")}
        if successful - set(expected):
            raise ProtocolError("journal result IDs are outside the frozen tasks")
        missing = tuple(identifier for identifier in expected if identifier not in successful)
        result = RunStatus(planned=len(expected), completed=len(successful), missing=missing,
                           state="partial" if state == "complete" and missing else state, records=records)
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
