"""Subprocess transport without experiment or model dependencies."""

import subprocess


def run_jsonl(binary, line, *, timeout=None):
    """Execute a one-request JSONL adapter and return stdout."""
    return subprocess.run([str(binary)], input=line, capture_output=True, text=True, check=True, timeout=timeout).stdout


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
