"""Run persistent SMP jobs with resumable records and graceful cancellation."""

import json
import os
import re
import signal
import subprocess
from pathlib import Path

from common.experiments.plans import freeze_manifest
from common.io.json import canonical
from common.runtime.process import cancellation_event


def _stop(process):
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _validate_summary(summary, metadata, steps, directory, exposure_trace):
    points = summary["endpoints"]
    if (
        summary["stop_reason"] != "horizon"
        or summary["stopped_at"] != steps[-1]
        or [point["step"] for point in points] != steps
        or [point["relative_step"] for point in points] != [step - steps[0] for step in steps]
        or len(summary["final_opinions"]) != metadata["NodeCount"]
    ):
        raise ValueError(f"incomplete persistent observation window: {directory}")
    previous = steps[0]
    for point in points:
        step = point["step"]
        if point["exposure"]["updates"] != (step - previous) * metadata["NodeCount"]:
            raise ValueError(f"invalid persistent exposure denominator: {directory}")
        previous = step
    if not (directory / "trajectory.json").is_file() or not any(directory.glob("finished*")):
        raise ValueError(f"missing completed SMP record: {directory}")
    if exposure_trace is not None:
        receipt = summary["exposure_trace"]
        if (
            Path(receipt["path"]).resolve() != exposure_trace
            or not exposure_trace.is_file()
            or receipt["updates"] != sum(point["exposure"]["updates"] for point in points)
            or receipt["slots"] != sum(point["exposure"]["recommended"] for point in points)
        ):
            raise ValueError(f"invalid persistent exposure trace receipt: {directory}")


def observe_persistent(
    binary, base_path, metadata, steps, *, policy_change=None, fixed_horizon=True, exposure_trace=None
):
    """Execute one native ``smp --job`` and return its online observations.

    The job and full raw history live in ``base_path``. Repeating the exact job
    resumes its latest physical checkpoint, including policy and observation
    state. Independent paired arms use the same initial RNG and replay their
    common prefix. Worker cancellation signals SMP and waits for its checkpoint.
    """
    binary, base_path = Path(binary).resolve(), Path(base_path).resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if not os.access(binary, os.X_OK):
        raise PermissionError(f"binary is not executable: {binary}")
    name = metadata.get("UniqueName")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", name) or name in (".", ".."):
        raise ValueError("persistent SMP job needs a valid UniqueName")
    steps = list(steps)
    if (
        not steps
        or any(type(step) is not int or step < 0 for step in steps)
        or any(left >= right for left, right in zip(steps, steps[1:]))
    ):
        raise ValueError("persistent observation steps must be increasing nonnegative integers")
    if type(fixed_horizon) is not bool:
        raise ValueError("fixed_horizon must be boolean")
    directory = base_path / name
    summary_path = directory / "observations.json"
    job = {
        "base_path": str(base_path),
        "metadata": metadata,
        "observe_at": steps,
        "fixed_horizon": fixed_horizon,
        "summary_path": str(summary_path),
    }
    if policy_change is not None:
        job["policy_change"] = policy_change
    if exposure_trace is not None:
        exposure_trace = Path(exposure_trace).resolve()
        job["exposure_trace"] = str(exposure_trace)
        job["trace_start"] = steps[0]
    # Canonical conversion also rejects nonfinite parameters before launching.
    job = json.loads(canonical(job))
    job_path = base_path / ".jobs" / (name + ".json")
    freeze_manifest(job_path, job)
    directory.mkdir(parents=True, exist_ok=True)
    stop = cancellation_event()
    if stop is not None and stop.is_set():
        raise KeyboardInterrupt("task execution cancelled")
    with (directory / "smp.log").open("ab") as log:
        process = subprocess.Popen(
            [str(binary), "--job", str(job_path)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=os.name == "posix",
        )
        try:
            while True:
                if stop is not None and stop.is_set():
                    raise KeyboardInterrupt("task execution cancelled")
                try:
                    returncode = process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if returncode == 130:
                raise KeyboardInterrupt(f"SMP job interrupted: {name}")
            if returncode:
                raise subprocess.CalledProcessError(returncode, process.args, stderr=f"see {directory / 'smp.log'}")
        except BaseException:
            _stop(process)
            raise
    summary = json.loads(summary_path.read_text())
    _validate_summary(summary, metadata, steps, directory, exposure_trace)
    summary["run_directory"] = str(directory)
    return summary
