"""Bounded, interruptible task execution; study definitions stay outside."""

import concurrent.futures
import math
import os
import signal
import subprocess
import threading
import time

from common.io.json import canonical, fingerprint, read_result, save_result, sha
from common.runtime.environment import runtime_environment

from .protocol import read_response, task_line
from .store import result_path

STOP = threading.Event()
ACTIVE = set()
ACTIVE_LOCK = threading.Lock()


def signal_group(pid, sig):
    """Signal a subprocess group, tolerating a process that completed before the signal arrived."""
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        # A completed process can disappear between polling and signalling.
        pass


def cancel(signum, frame):
    """Set the shared stop event and terminate active subprocess groups; used as SIGINT/SIGTERM handler."""
    STOP.set()
    with ACTIVE_LOCK:
        for pid in list(ACTIVE):
            signal_group(pid, signal.SIGTERM)


def execute(job, binary, build_id, label, deadline, job_timeout, outputs):
    """Run one JSONL task or reuse its identically bound saved result.

    deadline is an absolute monotonic time; job_timeout is per-task seconds and
    zero disables it. Return a status record distinguishing budget exhaustion,
    interrupts, timeouts, engine failures and completion. Save successful results
    atomically with their original job and build identity.
    """
    path = result_path(job, label, outputs)
    identity = fingerprint({"task": job, "build": build_id})
    if path.exists():
        old = read_result(path)
        if old["identity"] != identity:
            raise RuntimeError(f"incompatible result {path}")
        return {"id": job["id"], "status": "cached", "seconds": 0, "path": str(path)}
    started = time.monotonic()
    timeout = min(job_timeout if job_timeout else math.inf, deadline - started)
    if timeout <= 0:
        return {"id": job["id"], "status": "budget_unstarted", "seconds": 0}
    p = subprocess.Popen(
        [str(binary)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=runtime_environment(outputs.root),
        start_new_session=True,
    )
    with ACTIVE_LOCK:
        ACTIVE.add(p.pid)
        if STOP.is_set():
            signal_group(p.pid, signal.SIGTERM)
    try:
        stdout, stderr = p.communicate(task_line(job["payload"]), timeout=timeout if math.isfinite(timeout) else None)
    except subprocess.TimeoutExpired:
        signal_group(p.pid, signal.SIGTERM)
        try:
            stdout, stderr = p.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            signal_group(p.pid, signal.SIGKILL)
            stdout, stderr = p.communicate()
        status = "budget_interrupted" if time.monotonic() >= deadline else "job_timeout"
        with ACTIVE_LOCK:
            ACTIVE.discard(p.pid)
        return {"id": job["id"], "status": status, "seconds": time.monotonic() - started, "stderr": stderr[-4000:]}
    with ACTIVE_LOCK:
        ACTIVE.discard(p.pid)
    if p.returncode:
        return {
            "id": job["id"],
            "status": "cancelled" if STOP.is_set() else "failed",
            "exit_code": p.returncode,
            "seconds": time.monotonic() - started,
            "stderr": stderr[-4000:],
        }
    try:
        value = read_response(stdout, job["id"])
    except (ValueError, KeyError, RuntimeError) as error:
        return {"id": job["id"], "status": "failed", "seconds": time.monotonic() - started, "stderr": str(error)}
    record = {
        "identity": identity,
        "job": job,
        "build_fingerprint": build_id,
        "seconds": time.monotonic() - started,
        "result": value,
    }
    save_result(path, record)
    return {"id": job["id"], "status": "complete", "seconds": record["seconds"], "path": str(path), "sha256": sha(path)}


def run_group(jobs, binary, build_id, label, deadline, workers, job_timeout, journal, outputs):
    """Execute a family with bounded workers, journal terminal statuses and mark jobs not reached before deadline/stop as budget_unstarted."""
    pending = iter(jobs)
    active = {}
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:

        def submit():
            if time.monotonic() >= deadline or STOP.is_set():
                return False
            try:
                job = next(pending)
            except StopIteration:
                return False
            print(f"start {job['family']} {job['id']} {job['tier']}", flush=True)
            future = pool.submit(execute, job, binary, build_id, label, deadline, job_timeout, outputs)
            active[future] = job
            return True

        for _ in range(workers):
            submit()
        while active:
            done, _ = concurrent.futures.wait(active, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
            if not done:
                print(
                    f"heartbeat: active={len(active)} "
                    + (
                        f"remaining={max(0, deadline - time.monotonic()):.0f}s"
                        if math.isfinite(deadline)
                        else "wall budget=unlimited"
                    ),
                    flush=True,
                )
                continue
            for future in done:
                job = active.pop(future)
                try:
                    record = future.result()
                except Exception as e:
                    record = {"id": job["id"], "status": "failed", "error": str(e)}
                records.append(record)
                with journal.open("a") as f:
                    f.write(canonical(record) + "\n")
                    f.flush()
                print(f"done {job['family']} {job['id']} {record['status']}", flush=True)
                submit()
    reached = {r["id"] for r in records}
    return records + [{"id": j["id"], "status": "budget_unstarted"} for j in jobs if j["id"] not in reached]
