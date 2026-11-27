"""Stage-checkpoint and online-timing intervention scheduling."""

from pathlib import Path
from dataclasses import replace
from ehk.domain import (Plan, PolicyChange, FullResult, StagedResult, StageResult, Binding,
                        StageEvidence, DeferredRun, CampaignStatus)

from common.experiments.lifecycle import interruptible, session_lock
from common.experiments.plans import freeze_manifest
from common.io.json import sha
from ehk.metrics.pathway import EXCESS_THRESHOLD

from ..recommendation_shared import runtime
from . import design as cfg


def make_plan(study):
    if study not in cfg.STUDIES:
        raise ValueError(f"unknown intervention study: {study}")
    return Plan(schema=cfg.DESIGN_VERSION, study="paper_recommendation", profile=study,
                protocol=cfg.design_protocol(study), tasks=tuple(cfg.generate_tasks(study)))


def run_task(study, binary, directory, task, *, existing=None, on_progress=None):
    if study not in cfg.STUDIES:
        raise ValueError(f"unknown intervention study: {study}")
    switch = PolicyChange(recommender=cfg.SWITCH, at=0)

    def publish(result):
        if on_progress is not None:
            on_progress(result)

    if study == "staged_timing":
        return FullResult(arms=runtime.paired(
            binary, directory, task, "full", 0, cfg.REFERENCE_HORIZON,
            {
                "hold": None, "always_structure": switch,
                "fixed": replace(switch, at=cfg.FIXED_TRIGGER, duration=cfg.FOLLOWUP),
                "state": PolicyChange(recommender=cfg.SWITCH,excess_at_least=EXCESS_THRESHOLD,
                    latest=cfg.LATEST_TRIGGER,duration=cfg.FOLLOWUP),
            }, observation_steps=cfg.observation_steps, existing=existing.arms if existing is not None else None,
            on_progress=lambda arms: publish(FullResult(arms=arms)),
        ))
    reference = existing.reference if existing is not None else None
    if reference is None or not runtime.complete_raw(directory, reference):
        reference = runtime.observe(
            binary, directory, task, "reference", "reference", cfg.observation_steps(cfg.REFERENCE_HORIZON),
            checkpoints=cfg.STAGE_CHECKPOINTS,
        )
    phases = dict(existing.stages) if existing is not None else {}
    selected = cfg.select_stages(reference.endpoints)
    if phases.keys() - selected.keys():
        raise ValueError("stored staged progress has checkpoints outside the reference selection")
    publish(StagedResult(reference=reference,stages=phases))
    for phase, point in selected.items():
        step = point.step
        change = replace(switch,at=step)
        arms = {"hold": None, "switch": change}
        if phase == "late":
            arms["bridge"] = replace(change,bridge=True)
            if task.metadata.HKParams.RepostRate > 0:
                arms["hide_reposts"] = replace(change,hide_reposts=True)
        previous = phases.get(phase)
        if previous and previous.checkpoint != point:
            raise ValueError("stored staged progress differs from its selected reference checkpoint")

        def save_arms(observed, phase=phase, point=point):
            phases[phase] = StageResult(checkpoint=point,arms=observed)
            publish(StagedResult(reference=reference,stages=phases))

        phases[phase] = StageResult(checkpoint=point,arms=runtime.paired(
                binary, directory, task, phase, step, cfg.FOLLOWUP, arms,
                source=runtime.checkpoint_path(directory, task, "reference", "reference", step),
                detailed=phase == "late", observation_steps=cfg.observation_steps,
                existing=previous.arms if previous is not None else None, on_progress=save_arms,
            ))
    return StagedResult(reference=reference,stages=phases)


def run_study(study, directory, binary, workers, *, stage_evidence=None):
    directory, binary = Path(directory).resolve(), Path(binary).resolve()
    if not 1 <= workers <= 6:
        raise ValueError("recommendation studies support 1 through 6 concurrent simulation processes")
    if study not in cfg.STUDIES:
        raise ValueError(f"unknown intervention study: {study}")
    if study == "staged_timing":
        from .analysis import read_study, stage_gate

        source = Path(stage_evidence).resolve() if stage_evidence else Path(directory).resolve().parent / "staged"
        source_plan, records = read_study(source, "staged")
        gate = stage_gate(source_plan.tasks, records)
        if not gate.passed:
            raise ValueError(f"stage contrast gate did not pass: {gate}")
        evidence = StageEvidence(plan=source_plan.plan_fingerprint,contrast=gate)
    else:
        evidence = None

    def result_arms(result):
        if study == "staged":
            return [result.reference, *(arm for stage in result.stages.values() for arm in stage.arms.values())]
        return result.arms.values()

    return runtime.run_dataset(
        directory, binary, workers, plan_factory=lambda: make_plan(study),
        binding_factory=lambda plan, path, binary_sha: Binding(binary_sha256=binary_sha,stage_evidence=evidence),
        task_runner=lambda task, plan, binding, previous, progress: run_task(
            study, binary, Path(directory).resolve(), task, existing=previous, on_progress=progress,
        ),
        result_arms=result_arms, resume=True,
    )


def _preflight(binary, control):
    runtime.storage_preflight(binary, control, cfg.generate_tasks("staged")[0].metadata)


def run_campaign(directory, binary, workers):
    """Run the staged study and release timing only after its evidence gate."""
    from common.io.json import atomic_json

    from .analysis import read_study, stage_gate

    directory = Path(directory).resolve()
    if not 1 <= workers <= 6:
        raise ValueError("recommendation studies support 1 through 6 concurrent simulation processes")
    control = directory / "recommendation_runtime"
    with session_lock(control), interruptible():
        freeze_manifest(control / "binding.json", Binding(binary_sha256=sha(binary)))
        outcomes = {}
        try:
            atomic_json(control / "status.json", CampaignStatus(state="running",action="preflight",results=outcomes))
            _preflight(binary, control)
            atomic_json(control / "status.json", CampaignStatus(state="running",study="staged",results=outcomes))
            outcomes["staged"] = run_study("staged", directory / "staged", binary, workers)
            plan, records = read_study(directory / "staged", "staged")
            gate = stage_gate(plan.tasks, records)
            atomic_json(control / "timing_gate.json", gate)
            if gate.passed:
                outcomes["staged_timing"] = run_study("staged_timing", directory / "staged_timing", binary, workers)
            else:
                outcomes["staged_timing"] = DeferredRun(state="deferred",reason="stage evidence gate did not pass")
        except BaseException as error:
            atomic_json(control / "status.json", CampaignStatus(
                state="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=str(error),results=outcomes))
            raise
        state = "partial" if any(not isinstance(result,DeferredRun) and result.missing for result in outcomes.values()) else "complete"
        atomic_json(control / "status.json", CampaignStatus(state=state,results=outcomes))
        return outcomes
