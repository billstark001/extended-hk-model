"""Subprocess transport without experiment or model dependencies."""

import subprocess
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

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


def run_tasks(function, tasks, workers):
    """Bound queued work; on failure/interrupt stop launching and kill active solvers."""
    stop = threading.Event()

    def run(task):
        _local.stop = stop
        try:
            return function(task)
        finally:
            del _local.stop

    pool = ThreadPoolExecutor(max_workers=workers)
    iterator = iter(tasks)
    pending = set()
    try:
        for _ in range(workers):
            task = next(iterator, None)
            if task is not None:
                pending.add(pool.submit(run, task))
        while pending:
            complete, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in complete:
                future.result()
            for _ in complete:
                task = next(iterator, None)
                if task is not None:
                    pending.add(pool.submit(run, task))
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
