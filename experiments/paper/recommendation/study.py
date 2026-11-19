"""Run paired recommendation studies with complete, resumable SMP histories."""

from pathlib import Path

from common.experiments.execution import TaskAdapter, execute_tasks
from common.experiments.lifecycle import TaskJournal, Workflow, interruptible, session_lock
from common.experiments.plans import freeze_manifest, freeze_plan
from common.io.json import fingerprint, read_result, save_result, sha
from ehk.metrics.pathway import EXCESS_THRESHOLD
from ehk.micro.persistent import observe_persistent

from . import design as cfg


def make_plan(study):
    return {
        "schema": cfg.DESIGN_VERSION,
        "study": "paper_recommendation",
        "profile": study,
        "protocol": cfg.design_protocol(study),
        "tasks": cfg.generate_tasks(study),
    }


def _observe(binary, dataset, task, phase, arm, steps, *, change=None, detailed=False, checkpoints=()):
    metadata = {
        **task["metadata"], "UniqueName": arm, "MaxSimulationStep": steps[-1],
        "CheckpointSteps": sorted(set((*checkpoints, steps[0]))),
        "AgentNumber": True, "OpinionSum": True, "RewiringEvent": True, "PostEvent": True,
        "ViewPostsEvent": False, "ViewPostsDetails": False,
        "TrajectoryPrecision": {"opinions": "float64", "opinion_sums": "float64"},
    }
    parent = dataset / "raw" / task["id"] / phase
    trace = parent / arm / "exposure.jsonl.gz" if detailed else None
    result = observe_persistent(
        binary, parent, metadata, steps, policy_change=change,
        exposure_trace=trace, fixed_horizon=True,
    )
    result.pop("run_directory")
    result["raw_record"] = str((parent / arm).relative_to(dataset))
    if trace is not None:
        result["exposure_trace"]["path"] = str(trace.relative_to(dataset))
    return result


def _physical_checkpoint(path):
    from smp_bindings.model_state import load_snapshot

    return load_snapshot(str(path))


def _checkpoint_path(dataset, task, phase, arm, step):
    return dataset / "raw" / task["id"] / phase / arm / f"checkpoint-{step:09d}.msgpack.lz4"


def _paired(binary, dataset, task, phase, start, horizon, arms, *, source=None, detailed=False):
    steps = [start + step for step in cfg.observation_steps(horizon)]
    result = {}
    initial = _physical_checkpoint(source) if source is not None else None
    for arm, change in arms.items():
        observed = _observe(binary, dataset, task, phase, arm, steps, change=change, detailed=detailed)
        checkpoint = _physical_checkpoint(_checkpoint_path(dataset, task, phase, arm, start))
        if initial is None:
            initial = checkpoint
        if checkpoint != initial:
            raise ValueError(f"paired arms differ in microscopic state or RNG at step {start}: {task['id']}")
        if source is not None:
            observed["source_sha256"] = sha(source)
        result[arm] = observed
    return result


def run_task(study, binary, directory, task):
    if study == "recommender_variants":
        arms = {
            policy: None if policy == cfg.REFERENCE else {"recommender": policy, "at": 0}
            for family in cfg.VARIANTS.values() for policy in family
        }
        return {"arms": _paired(binary, directory, task, "full", 0, cfg.REFERENCE_HORIZON, arms)}
    switch = {"recommender": cfg.SWITCH, "at": 0}
    if study == "staged_timing":
        return {"arms": _paired(
            binary, directory, task, "full", 0, cfg.REFERENCE_HORIZON,
            {
                "hold": None, "always_structure": switch,
                "fixed": {**switch, "at": cfg.FIXED_TRIGGER, "duration": cfg.FOLLOWUP},
                "state": {
                    "recommender": cfg.SWITCH, "excess_at_least": EXCESS_THRESHOLD,
                    "latest": cfg.LATEST_TRIGGER, "duration": cfg.FOLLOWUP,
                },
            },
        )}
    reference = _observe(
        binary, directory, task, "reference", "reference", cfg.observation_steps(cfg.REFERENCE_HORIZON),
        checkpoints=cfg.STAGE_CHECKPOINTS,
    )
    phases = {}
    for phase, point in cfg.select_stages(reference["endpoints"]).items():
        step = point["step"]
        change = {**switch, "at": step}
        arms = {"hold": None, "switch": change}
        if phase == "late":
            arms["bridge"] = {**change, "bridge": True}
            if task["metadata"]["HKParams"]["RepostRate"] > 0:
                arms["hide_reposts"] = {**change, "hide_reposts": True}
        phases[phase] = {
            "checkpoint": point,
            "arms": _paired(
                binary, directory, task, phase, step, cfg.FOLLOWUP, arms,
                source=_checkpoint_path(directory, task, "reference", "reference", step),
                detailed=phase == "late",
            ),
        }
    return {"reference": reference, "stages": phases}


def _raw_complete(directory, arm):
    raw = directory / arm["raw_record"]
    if not all((raw / name).is_file() for name in ("metadata.json", "trajectory.json", "events.db", "observations.json")):
        return False
    if not any(raw.glob("finished*")):
        return False
    trace = arm.get("exposure_trace")
    return trace is None or (directory / trace["path"]).is_file()


def run_study(study, directory, binary, workers, *, stage_evidence=None):
    directory, binary = Path(directory).resolve(), Path(binary).resolve()
    if not 1 <= workers <= 6:
        raise ValueError("recommendation studies support 1 through 6 concurrent simulation processes")
    if study == "staged_timing":
        from .analysis import read_study, stage_gate

        source = Path(stage_evidence).resolve() if stage_evidence else directory.parent / "staged"
        source_plan, records = read_study(source, "staged")
        gate = stage_gate(source_plan["tasks"], records)
        if not gate["passed"]:
            raise ValueError(f"stage contrast gate did not pass: {gate}")
        evidence = {"plan": source_plan["plan_fingerprint"], "contrast": gate}
    else:
        evidence = None

    control = directory / "runtime"
    with session_lock(control), interruptible():
        workflow = Workflow(
            {"plan": lambda: freeze_plan(directory / "plan.json", make_plan(study))}, control / "attempt.json"
        )
        plan = workflow.dispatch("plan")
        freeze_manifest(control / "binding.json", {"binary_sha256": sha(binary), "stage_evidence": evidence})
        journal = TaskJournal(control)
        tasks = plan["tasks"]

        def result_path(task):
            return directory / "records" / (task["id"] + ".json.gz")

        def cached(task):
            path = result_path(task)
            if not path.is_file():
                return False
            record = read_result(path)
            if record["task_fingerprint"] != fingerprint(task):
                return False
            result = record["result"]
            arms = (
                [result["reference"], *(arm for stage in result["stages"].values() for arm in stage["arms"].values())]
                if study == "staged" else list(result["arms"].values())
            )
            return all(_raw_complete(directory, arm) for arm in arms)

        def save(task, result):
            save_result(result_path(task), {"task_fingerprint": fingerprint(task), "result": result})

        def execute():
            state = "complete"
            try:
                pending = []
                for task in tasks:
                    if cached(task):
                        journal.event(task["id"], "cached")
                    else:
                        pending.append(task)
                adapter = TaskAdapter(
                    lambda task: task["id"], cached, lambda task: run_task(study, binary, directory, task), save
                )
                execute_tasks(pending, adapter, workers=workers, journal=journal, raise_failures=False)
            except BaseException as error:
                state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                raise
            finally:
                result = journal.finish(tasks, state)
            return result

        workflow.stages["run"] = execute
        return workflow.dispatch("run")


def _preflight(binary, control):
    """Check native persistence on the selected volume before scheduling a design."""
    metadata = {
        **cfg.generate_tasks("staged")[0]["metadata"],
        "UniqueName": "storage-check",
        "NodeCount": 8,
        "NodeFollowCount": 2,
        "RecsysCount": 3,
        "MaxSimulationStep": 2,
        "CheckpointSteps": [0, 2],
        "AgentNumber": True,
        "OpinionSum": True,
        "RewiringEvent": True,
        "PostEvent": True,
        "ViewPostsEvent": False,
        "ViewPostsDetails": False,
        "TrajectoryPrecision": {"opinions": "float64", "opinion_sums": "float64"},
    }
    for _ in range(2):
        observe_persistent(binary, control / "preflight", metadata, [0, 2], fixed_horizon=True)


def run_campaign(directory, binary, workers):
    """Run both independent designs and release timing only after its evidence gate."""
    from common.io.json import atomic_json

    from .analysis import read_study, stage_gate

    directory = Path(directory).resolve()
    if not 1 <= workers <= 6:
        raise ValueError("recommendation studies support 1 through 6 concurrent simulation processes")
    control = directory / "recommendation_runtime"
    with session_lock(control), interruptible():
        freeze_manifest(control / "binding.json", {"binary_sha256": sha(binary)})
        outcomes = {}
        try:
            atomic_json(control / "status.json", {"state": "running", "action": "preflight", "results": outcomes})
            _preflight(binary, control)
            for study in ("staged", "recommender_variants"):
                atomic_json(control / "status.json", {"state": "running", "study": study, "results": outcomes})
                outcomes[study] = run_study(study, directory / study, binary, workers)
            plan, records = read_study(directory / "staged", "staged")
            gate = stage_gate(plan["tasks"], records)
            atomic_json(control / "timing_gate.json", gate)
            if gate["passed"]:
                outcomes["staged_timing"] = run_study("staged_timing", directory / "staged_timing", binary, workers)
            else:
                outcomes["staged_timing"] = {"state": "deferred", "reason": "stage evidence gate did not pass"}
        except BaseException as error:
            atomic_json(control / "status.json", {
                "state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error": str(error), "results": outcomes,
            })
            raise
        state = "partial" if any(result.get("missing") for result in outcomes.values()) else "complete"
        atomic_json(control / "status.json", {"state": state, "results": outcomes})
        return outcomes
