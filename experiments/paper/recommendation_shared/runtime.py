"""Persistent paired execution, independent of any scientific study protocol."""

from pathlib import Path
import sys

from common.experiments.progress import Progress
from dataclasses import replace

from common.experiments.execution import TaskAdapter, execute_tasks
from common.experiments.lifecycle import RunStatus, TaskJournal, Workflow, interruptible, session_lock
from common.experiments.plans import freeze_manifest
from ehk.domain import (Binding, FullResult, Plan, PolicyChange, RecommenderParams, ScenarioMetadata,
                        Task, TrajectoryPrecision, freeze_plan)
from common.io.json import sha
from ehk.micro.persistent import observe_persistent
from .storage import StudyStore, raw_availability


def observe(
    binary, dataset, task, phase, arm, steps, *, change=None, detailed=False, checkpoints=(), parameters=None,
):
    metadata = replace(task.metadata, UniqueName=arm, MaxSimulationStep=steps[-1],
                       CheckpointSteps=tuple(sorted(set((*checkpoints, steps[0])))),
                       AgentNumber=True, OpinionSum=True, RewiringEvent=True, PostEvent=True,
                       ViewPostsEvent=False, ViewPostsDetails=False,
                       TrajectoryPrecision=TrajectoryPrecision(opinions="float64", opinion_sums="float64"))
    if parameters is not None:
        metadata = replace(metadata, RecSysParams=(metadata.RecSysParams or RecommenderParams()).with_overrides(parameters))
    parent = dataset / "raw" / task.id / phase
    trace = parent / arm / "exposure.jsonl.gz" if detailed else None
    result = observe_persistent(
        binary, parent, metadata, steps, policy_change=change,
        exposure_trace=trace, fixed_horizon=True,
    )
    result.run_directory = None
    result.raw_record = (parent / arm).relative_to(dataset).as_posix()
    if trace is not None:
        result.exposure_trace.path = trace.relative_to(dataset).as_posix()
    return result


def physical_checkpoint(path):
    from smp_bindings.model_state import load_snapshot

    return load_snapshot(str(path))


def checkpoint_path(dataset, task, phase, arm, step):
    return dataset / "raw" / task.id / phase / arm / f"checkpoint-{step:09d}.msgpack.lz4"


def paired(
    binary, dataset, task, phase, start, horizon, arms, *, source=None, detailed=False,
    parameters=None, existing=None, on_progress=None, binary_sha256=None, arm_binary_sha256=None,
    observation_steps,
):
    steps = [start + step for step in observation_steps(horizon)]
    result = dict(existing or {})
    if result.keys() - arms.keys():
        raise ValueError(f"record contains arms outside the frozen protocol: {task.id}")
    initial = physical_checkpoint(source) if source is not None else None
    for arm, change in arms.items():
        reusable = arm in result and complete_raw(dataset, result[arm])
        if reusable and arm_binary_sha256 is not None:
            expected = arm_binary_sha256[arm]
            if result[arm].binary_sha256 is not None and result[arm].binary_sha256 != expected:
                raise ValueError(f"retained arm binary provenance differs from its binding: {task.id}/{arm}")
        if not reusable and arm_binary_sha256 is not None and arm_binary_sha256[arm] != binary_sha256:
            raise ValueError(
                f"retained arm requires its original binary {arm_binary_sha256[arm]}: {task.id}/{arm}; "
                "restore its completed raw record before continuing with the new binary"
            )
        observed = result[arm] if reusable else observe(
            binary, dataset, task, phase, arm, steps, change=change, detailed=detailed,
            parameters=(parameters or {}).get(arm),
        )
        checkpoint = physical_checkpoint(checkpoint_path(dataset, task, phase, arm, start))
        if initial is None:
            initial = checkpoint
        if checkpoint != initial:
            raise ValueError(f"paired arms differ in microscopic state or RNG at step {start}: {task.id}")
        if source is not None:
            observed.source_sha256 = sha(source)
        result[arm] = observed
        if not reusable and binary_sha256 is not None:
            observed.binary_sha256 = binary_sha256
        if not reusable and on_progress is not None:
            on_progress(result)
    return result


def complete_raw(directory, arm):
    return raw_availability(directory, arm).status == "complete"


def storage_preflight(binary, control, task_metadata):
    """Check native persistence on the selected volume before scheduling a design."""
    metadata = replace(task_metadata, UniqueName="storage-check", NodeCount=8, NodeFollowCount=2,
                       RecsysCount=3, MaxSimulationStep=2, CheckpointSteps=(0, 2),
                       AgentNumber=True, OpinionSum=True, RewiringEvent=True, PostEvent=True,
                       ViewPostsEvent=False, ViewPostsDetails=False,
                       TrajectoryPrecision=TrajectoryPrecision(opinions="float64", opinion_sums="float64"))
    for _ in range(2):
        observe_persistent(binary, control / "preflight", metadata, [0, 2], fixed_horizon=True)


def run_dataset(directory, binary, workers, *, plan_factory, binding_factory, task_runner,
                result_arms, resume=False, cache_valid=None):
    """Schedule frozen paired tasks while each study owns its scientific callbacks."""
    directory, binary = Path(directory).resolve(), Path(binary).resolve()
    if not 1 <= workers <= 6:
        raise ValueError("recommendation studies support 1 through 6 concurrent simulation processes")
    control = directory / "runtime"
    with session_lock(control), interruptible():
        workflow = Workflow(
            {"plan": lambda: freeze_plan(directory / "plan.json", plan_factory())}, control / "attempt.json"
        )
        plan = workflow.dispatch("plan")
        binding = binding_factory(plan, control / "binding.json", sha(binary))
        freeze_manifest(control / "binding.json", binding)
        journal = TaskJournal(control)
        tasks = plan.tasks
        observations = StudyStore(directory)

        def cached(task):
            record = observations.get_task(task, plan)
            if record is None:
                return False
            result = record.result
            if cache_valid is not None and not cache_valid(result, binding):
                return False
            return all(complete_raw(directory, arm) for arm in result_arms(result))

        def save(task, result, *, complete=True):
            observations.put_task(task, result, plan=plan, binding=binding, complete=complete)

        def run(task):
            previous = None
            if resume:
                previous = observations.get_task(task, plan, pending=True)
                if previous is None:
                    previous = observations.get_task(task, plan)
            return task_runner(
                task, plan, binding, previous.result if previous is not None else None,
                lambda result: save(task, result, complete=False),
            )

        def execute() -> RunStatus:
            state = "complete"
            progress = Progress(len(tasks), stream=sys.stderr)
            try:
                pending = []
                for task in tasks:
                    if cached(task):
                        journal.event(task.id, "cached")
                        progress.event(task.id, "cached")
                    else:
                        pending.append(task)
                adapter = TaskAdapter(lambda task: task.id, cached, run, save)
                execute_tasks(pending, adapter, workers=workers, journal=journal, progress=progress, raise_failures=False)
            except BaseException as error:
                state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                raise
            finally:
                result = journal.finish((task.id for task in tasks), state)
            return result

        workflow.stages["run"] = execute
        try:
            return workflow.dispatch("run")
        finally:
            observations.close()
