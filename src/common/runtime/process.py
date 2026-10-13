"""Subprocess transport without experiment or model dependencies."""

import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .failures import BatchTaskError, ProtocolError

_local = threading.local()
_process_lock = threading.Lock()
_processes = {}


def run_jsonl(binary, line, *, timeout=None, arguments=()):
    """Execute a one-request JSONL adapter and return stdout."""
    stop = getattr(_local, "stop", None)
    with _process_lock:
        if stop is not None and stop.is_set():
            raise RuntimeError("task execution cancelled")
        process = subprocess.Popen(
            [str(binary), *arguments], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        _processes[process] = stop
    try:
        stdout, stderr = process.communicate(line, timeout=timeout)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, process.args, stdout, stderr)
        return stdout
    except BaseException:
        process.kill()
        process.communicate()
        raise
    finally:
        with _process_lock:
            _processes.pop(process, None)


def cancellation_event():
    return getattr(_local, "stop", None)


def run_binary(binary, line, output, *, arguments=(), timeout=None, max_bytes=1 << 30):
    """Drain binary stdout to APFS with bounded buffers; preserve cancellation."""
    stop = cancellation_event()
    with _process_lock:
        if stop is not None and stop.is_set():
            raise KeyboardInterrupt("task execution cancelled")
        process = subprocess.Popen(
            [str(binary), *arguments], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        _processes[process] = stop
    stderr = bytearray()

    def drain_errors():
        for block in iter(lambda: process.stderr.read(8192), b""):
            stderr.extend(block[: max(0, 65536 - len(stderr))])

    reader = threading.Thread(target=drain_errors, daemon=True)
    reader.start()
    timer = threading.Timer(timeout, process.kill) if timeout else None
    started = time.monotonic()
    if timer:
        timer.start()
    try:
        process.stdin.write(line.encode())
        process.stdin.close()
        count = 0
        with open(output, "wb") as stream:
            for block in iter(lambda: process.stdout.read(1 << 20), b""):
                count += len(block)
                if count > max_bytes:
                    raise RuntimeError("solver record exceeds configured local spool limit")
                stream.write(block)
        process.wait()
        reader.join()
        if timeout and time.monotonic() - started >= timeout and process.returncode:
            raise subprocess.TimeoutExpired(process.args, timeout)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, process.args, stderr=bytes(stderr))
    except BaseException:
        process.kill()
        process.wait()
        reader.join()
        raise
    finally:
        if timer:
            timer.cancel()
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
        with _process_lock:
            _processes.pop(process, None)


def run_tasks(function, tasks, workers, *, isolate_failures=False, attempts=3, event=None):
    """Bound workers; isolate ordinary failures and retry only transport failures.

    Scientific RuntimeError/ValueError are never retried. Frozen protocol errors
    and interrupts cancel all active solvers. Existing audit callers can retain
    fail-fast behavior; the main workflow explicitly opts into isolation.
    """
    stop = threading.Event()

    def run(task):
        _local.stop = stop
        try:
            for attempt in range(1, attempts + 1):
                if event:
                    event(task, "running", attempt=attempt)
                try:
                    result = function(task)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    if not isolate_failures or attempt == attempts:
                        raise
                    if event:
                        event(task, "retrying", attempt=attempt, error=str(error))
                    if stop.wait(min(2 ** (attempt - 1), 30)):
                        raise KeyboardInterrupt("task execution cancelled")
                else:
                    if event:
                        event(task, "complete", attempt=attempt)
                    return result
        finally:
            del _local.stop

    pool = ThreadPoolExecutor(max_workers=workers)
    iterator = iter(tasks)
    pending = set()
    submitted = {}
    failures = []
    try:
        for _ in range(workers):
            task = next(iterator, None)
            if task is not None:
                future = pool.submit(run, task)
                submitted[future] = task
                pending.add(future)
        while pending:
            complete, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in complete:
                task = submitted.pop(future)
                try:
                    future.result()
                except ProtocolError:
                    raise
                except Exception as error:
                    if not isolate_failures:
                        raise
                    failures.append(str(error))
                    if event:
                        event(task, "failed", error=str(error))
            for _ in complete:
                task = next(iterator, None)
                if task is not None:
                    future = pool.submit(run, task)
                    submitted[future] = task
                    pending.add(future)
        if failures:
            raise BatchTaskError(failures)
    except BaseException:
        with _process_lock:
            stop.set()
            for process, owner in list(_processes.items()):
                if owner is stop:
                    process.kill()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def acquire_lock(path):
    """Create an exclusive process lock containing the current PID."""
    import os

    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)
    except FileExistsError as error:
        raise RuntimeError(f"run already active; inspect PID in {path} before removing a stale lock") from error


def release_lock(path):
    """Remove a lock owned by the current workflow after its terminal cleanup."""
    path.unlink(missing_ok=True)
