"""Subprocess transport without experiment or model dependencies."""

import multiprocessing
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from typing import cast

from .failures import BatchTaskError, ProtocolError

_local = threading.local()
_process_lock = threading.Lock()
_processes = {}


def _isolated_call(connection, function, arguments):
    os.setsid()
    try:
        connection.send((True, function(*arguments)))
    except BaseException as error:
        connection.send((False, error))
    finally:
        connection.close()


def _kill_isolated_tree(pid):
    # A third-party launcher may start children in new sessions. Freeze the
    # launcher before collecting ancestry so those children cannot escape merely
    # by leaving its process group. This uses only OS process IDs, never API names.
    try:
        os.kill(pid, signal.SIGSTOP)
    except ProcessLookupError:
        pass
    descendants = {pid}
    try:
        result = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, check=True, timeout=2)
        relationships = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
        while True:
            children = {child for child, parent in relationships if parent in descendants}
            if children <= descendants:
                break
            descendants.update(children)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for child in descendants:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_isolated(function, *arguments):
    """Run an opaque callable in its own process group, cancelling its descendants.

    Callables and arguments must be spawn-picklable. This boundary lets adapters
    use third-party launchers without depending on their interruption handling.
    """
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_isolated_call, args=(child, function, arguments))
    try:
        process.start()
        child.close()
        stop = cancellation_event()
        while not parent.poll(0.1):
            if stop is not None and stop.is_set():
                raise KeyboardInterrupt("task execution cancelled")
            if not process.is_alive():
                raise RuntimeError(f"isolated task exited without a result ({process.exitcode})")
        success, result = parent.recv()
        if not success:
            raise result
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"isolated task exited with {process.exitcode}")
        return result
    except BaseException:
        if process.pid is not None:
            _kill_isolated_tree(process.pid)
            if process.is_alive():
                process.kill()
            process.join()
        raise
    finally:
        child.close()
        parent.close()
        if process.pid is not None:
            process.close()


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


def cancellation_event() -> threading.Event | None:
    return cast(threading.Event | None, getattr(_local, "stop", None))


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


def run_tasks(
    function,
    tasks,
    workers,
    *,
    isolate_failures=False,
    attempts=3,
    event=None,
    on_result=None,
    executor="thread",
    cancel=None,
    raise_failures=True,
):
    """Bound workers; isolate ordinary failures and retry only transport failures.

    Scientific RuntimeError/ValueError are never retried. Frozen protocol errors
    and interrupts cancel all active solvers. Existing audit callers can retain
    fail-fast behavior; the main workflow explicitly opts into isolation.
    """
    if workers < 1 or executor not in ("thread", "process"):
        raise ValueError("positive workers and thread/process executor required")
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

    pool = (ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor)(max_workers=workers)

    def submit(task):
        if executor == "process":
            if event:
                event(task, "running", attempt=1)
            return pool.submit(function, task)
        return pool.submit(run, task)

    iterator = iter(tasks)
    pending = set()
    submitted = {}
    failures = []
    try:
        for _ in range(workers):
            task = next(iterator, None)
            if task is not None:
                future = submit(task)
                submitted[future] = task
                pending.add(future)
        while pending:
            complete, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in complete:
                task = submitted.pop(future)
                try:
                    result = future.result()
                    if on_result is not None:
                        on_result(task, result)
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
                    future = submit(task)
                    submitted[future] = task
                    pending.add(future)
        if failures and raise_failures:
            raise BatchTaskError(failures)
    except BaseException:
        stop.set()
        if cancel is not None:
            cancel()
        if executor == "process":
            # Python 3.10-3.13 have no public terminate_workers API.
            children = list(getattr(pool, "_processes", {}).values())
            for child in children:
                if child.is_alive():
                    child.terminate()
            for child in children:
                child.join(timeout=0.2)
                if child.is_alive():
                    child.kill()
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
