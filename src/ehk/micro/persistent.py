"""Run persistent SMP jobs with resumable records and graceful cancellation."""

import json
import os
import re
import subprocess
from pathlib import Path

from common.experiments.plans import freeze_manifest
from common.runtime.process import cancellation_event
from ehk.domain import ArmObservation, PersistentJob, PolicyChange, ResearchFrame, ScenarioMetadata, parse_dataclass


def _stop(process, job_path):
    if process.poll() is not None:
        process.communicate()
        return
    Path(str(job_path) + ".cancel").touch()
    try:
        process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


def _validate_summary(summary: ArmObservation, metadata: ScenarioMetadata, steps, directory, exposure_trace):
    points = summary.endpoints
    if (
        summary.stop_reason != "horizon"
        or summary.stopped_at != steps[-1]
        or [point.step for point in points] != steps
        or [point.relative_step for point in points] != [step - steps[0] for step in steps]
        or len(summary.final_opinions) != metadata.NodeCount
    ):
        raise ValueError(f"incomplete persistent observation window: {directory}")
    previous = steps[0]
    for point in points:
        step = point.step
        if point.exposure.updates != (step - previous) * metadata.NodeCount:
            raise ValueError(f"invalid persistent exposure denominator: {directory}")
        previous = step
    if not (directory / "trajectory.json").is_file() or not any(directory.glob("finished*")):
        raise ValueError(f"missing completed SMP record: {directory}")
    if exposure_trace is not None:
        receipt = summary.exposure_trace
        if receipt is None:
            raise ValueError("native observer omitted the requested exposure receipt")
        if (
            Path(receipt.path).resolve() != exposure_trace
            or not exposure_trace.is_file()
            or receipt.updates != sum(point.exposure.updates for point in points)
            or receipt.slots != sum(point.exposure.recommended for point in points)
        ):
            raise ValueError(f"invalid persistent exposure trace receipt: {directory}")


def observe_persistent(
    binary, base_path, metadata: ScenarioMetadata, steps, *, policy_change: PolicyChange | None = None,
    fixed_horizon=True, exposure_trace=None
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
    name = metadata.UniqueName
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
    if exposure_trace is not None:
        exposure_trace = Path(exposure_trace).resolve()
    job = PersistentJob(base_path=str(base_path), metadata=metadata, observe_at=tuple(steps),
                        fixed_horizon=fixed_horizon, policy_change=policy_change,
                        exposure_trace=str(exposure_trace) if exposure_trace else None,
                        trace_start=steps[0] if exposure_trace else None)
    job_path = base_path / ".jobs" / (name + ".json")
    freeze_manifest(job_path, job)
    cancel_path = Path(str(job_path) + ".cancel")
    cancel_path.unlink(missing_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    stop = cancellation_event()
    if stop is not None and stop.is_set():
        raise KeyboardInterrupt("task execution cancelled")
    with (directory / "smp.log").open("ab") as log:
        process = subprocess.Popen(
            [str(binary), "--job", str(job_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=log,
            start_new_session=os.name == "posix",
            text=True,
            encoding="utf-8",
        )
        try:
            while True:
                if stop is not None and stop.is_set():
                    raise KeyboardInterrupt("task execution cancelled")
                try:
                    output, _ = process.communicate(timeout=0.1)
                    returncode = process.returncode
                    break
                except subprocess.TimeoutExpired:
                    pass
            if returncode == 130:
                raise KeyboardInterrupt(f"SMP job interrupted: {name}")
            if returncode:
                raise subprocess.CalledProcessError(returncode, process.args, stderr=f"see {directory / 'smp.log'}")
        except BaseException:
            _stop(process, job_path)
            raise
    message = parse_dataclass(ResearchFrame, json.loads(output))
    if message.type != "research_result" or message.protocol != "smp-research-result-v1":
        raise ValueError("native research result has an invalid protocol")
    summary = message.result
    _validate_summary(summary, metadata, steps, directory, exposure_trace)
    summary.run_directory = str(directory)
    return summary
