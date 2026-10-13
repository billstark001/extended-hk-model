"""APFS result staging and cancellable recovery from external-storage outages."""

import errno
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

from common.io.json import atomic_json, fingerprint, sha

from .failures import ProtocolError
from .process import cancellation_event

STORAGE_ERRORS = {errno.EIO, errno.ENXIO, errno.ENODEV, errno.ESTALE, errno.ENOSPC}


def storage_error(error):
    return isinstance(error, OSError) and error.errno in STORAGE_ERRORS


def retry_io(operation, *, event=None, delay=1.0):
    """An unreadable disk is not a corrupt record. Wait without resampling seeds."""
    attempt = 0
    while True:
        try:
            return operation()
        except OSError as error:
            if not storage_error(error):
                raise
            attempt += 1
            if event:
                event(None, "waiting_for_storage", error=str(error), attempt=attempt)
            stop = cancellation_event()
            pause = min(delay * 2 ** min(attempt - 1, 5), 30)
            if stop:
                if stop.wait(pause):
                    raise KeyboardInterrupt("storage wait cancelled")
            else:
                time.sleep(pause)


class RunStorage:
    """One local spool per frozen run. Commit ready records before scheduling more.

    The caller bounds simultaneous tasks; each binary response is additionally
    capped at max_record_bytes. Pending commits survive launcher restarts.
    """

    def __init__(self, root, control, *, max_record_bytes=512 << 20):
        self.root = Path(root).resolve()
        self.control = Path(control).resolve()
        self.spool = self.control / "spool"
        self.spool.mkdir(parents=True, exist_ok=True)
        self.max_record_bytes = max_record_bytes
        self._journal_lock = threading.Lock()
        self.marker = self.root / "storage_owner.json"
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        owner = {"schema": 1, "root": str(self.root), "control": str(self.control)}
        if self.marker.exists():
            if json.loads(self.marker.read_text()) != owner:
                raise ProtocolError("storage ownership changed")
        else:
            atomic_json(self.marker, owner)
        self.owner = owner

    def event(self, task, state, **extra):
        identifier = None
        if task is not None:
            item = task[1] if isinstance(task, tuple) else task
            identifier = item.get("id") if isinstance(item, dict) else str(item)
        with self._journal_lock, (self.control / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": time.time(), "task": identifier, "state": state, **extra}) + "\n")
            stream.flush()

    def check(self):
        try:
            saved = json.loads(self.marker.read_text())
        except FileNotFoundError as error:
            raise OSError(errno.ENODEV, "run volume or ownership marker unavailable", str(self.root)) from error
        if saved != self.owner:
            raise ProtocolError("a different volume/run occupies the result path")

    def read(self, operation):
        def checked():
            self.check()
            return operation()

        return retry_io(checked, event=self.event)

    def temporary(self, prefix="transport-"):
        fd, path = tempfile.mkstemp(prefix=prefix, dir=self.spool)
        os.close(fd)
        return Path(path)

    def commit(self, destination, writer):
        destination = Path(destination).resolve()
        if not destination.is_relative_to(self.root):
            raise ProtocolError("result destination escapes its run")
        relative = str(destination.relative_to(self.root))
        key = fingerprint({"destination": relative})
        source = self.spool / (key + destination.suffix)
        pending = self.spool / (key + ".pending.json")
        if not pending.exists():
            atomic_json(pending, {"destination": relative, "source": source.name, "sha256": None})
        if not source.exists():
            writer(source)
        metadata = json.loads(pending.read_text())
        metadata["sha256"] = sha(source)
        atomic_json(pending, metadata)
        self.publish(pending)

    def publish(self, pending):
        metadata = json.loads(pending.read_text())
        source = self.spool / metadata["source"]
        destination = self.root / metadata["destination"]
        if not destination.resolve().is_relative_to(self.root) or source.parent != self.spool:
            raise ProtocolError("invalid staged result destination")
        expected = metadata["sha256"] or sha(source)

        def copy():
            self.check()
            if destination.exists() and sha(destination) == expected:
                return
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".upload.tmp")
            try:
                with source.open("rb") as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
                    dst.flush()
                    os.fsync(dst.fileno())
                if sha(temporary) != expected:
                    raise OSError(errno.EIO, "external copy checksum mismatch")
                os.replace(temporary, destination)
                if sha(destination) != expected:
                    raise OSError(errno.EIO, "committed record checksum mismatch")
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass  # Next attempt replaces this task's incomplete upload.

        retry_io(copy, event=self.event)
        self.event(None, "record_committed", path=str(destination), sha256=expected)
        pending.unlink()
        source.unlink()

    def recover(self):
        for pending in sorted(self.spool.glob("*.pending.json")):
            metadata = json.loads(pending.read_text())
            if (self.spool / metadata["source"]).exists():
                self.publish(pending)
            else:
                pending.unlink()  # Writer never completed its atomic local record.
