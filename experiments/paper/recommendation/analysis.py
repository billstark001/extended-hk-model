"""Analyze matched interventions using typed observations and summaries."""

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from ehk.domain import Endpoint, MeanInterval, StageGate, Task, rows_frame
from experiments.paper.recommendation_shared import analysis as shared
from . import design as cfg


@dataclass(frozen=True)
class InterventionContrast:
    treatment: str
    control: str


CONTRASTS = {
    "staged": (InterventionContrast("switch", "hold"), InterventionContrast("bridge", "switch"),
               InterventionContrast("hide_reposts", "switch")),
    "staged_timing": (InterventionContrast("state", "fixed"), InterventionContrast("state", "hold"),
                      InterventionContrast("state", "always_structure")),
}


@dataclass(frozen=True)
class ReferenceSample:
    task: Task
    reference_outcome: str
    point: Endpoint


@dataclass(frozen=True)
class SelectionSample:
    task: Task
    reference_outcome: str
    stage: str
    point: Endpoint


@dataclass(kw_only=True)
class StageInteraction(MeanInterval):
    sensitivity: str
    epsilon: float
    history: int
    setting: str
    repost: float | str
    earlier: str
    later: str
    metric: str


def read_study(directory, study=None):
    return shared.read_study(directory, study, profiles=cfg.STUDIES)


def completed_task_ids(plan, records):
    if plan.profile not in cfg.STUDIES:
        raise ValueError(f"not an intervention profile: {plan.profile}")
    return {task.id for task in plan.tasks if task.id in records}


def study_tables(plan, records, *, binding=None):
    if plan.profile not in cfg.STUDIES:
        raise ValueError(f"not an intervention profile: {plan.profile}")
    return shared.study_tables(plan, records, binding=binding)


def coverage_by_setting(plan, records):
    return shared.coverage_by_setting(
        plan, records, completed=completed_task_ids(plan, records),
        stages=cfg.STAGES if plan.profile == "staged" else (),
    )


def completed_cell_task_ids(plan, records):
    return shared.completed_cell_task_ids(plan, records, coverage=coverage_by_setting(plan, records))


def contrast_table(runs, study):
    return shared.contrast_table(runs, tuple((pair.treatment, pair.control) for pair in CONTRASTS[study]))


def stage_gate(tasks, records) -> StageGate:
    core = [task for task in tasks if task.sensitivity == "core"]
    differences = []
    for task in core:
        if task.id not in records:
            continue
        phases = records[task.id].stages
        if "early" in phases and "late" in phases:
            effects = []
            for phase in ("early", "late"):
                arms = phases[phase].arms
                effects.append(arms["switch"].endpoints[-1].pathway.separation
                               - arms["hold"].endpoints[-1].pathway.separation)
            differences.append(effects[1] - effects[0])
    estimate = shared.mean_interval(differences)
    completed = sum(task.id in records for task in core)
    complete = bool(core) and completed == len(core)
    passed = bool(complete and estimate.lo is not None and estimate.hi is not None
                  and abs(estimate.mean) >= cfg.MIN_STAGE_EFFECT and (estimate.lo > 0 or estimate.hi < 0))
    return StageGate(n=estimate.n, mean=estimate.mean, lo=estimate.lo, hi=estimate.hi,
                     planned_core=len(core), completed_core=completed,
                     complete=complete, passed=passed, minimum_effect=cfg.MIN_STAGE_EFFECT)


COORDINATE_COLUMNS = ("task_id", "setting", "sensitivity", "epsilon", "history",
                      "influence", "rewiring", "repost", "replicate")
PATHWAY_COLUMNS = ("separation", "pair_concordance", "edge_concordance", "excess_concordance")


def _coordinate_values(task):
    metadata = task.metadata
    return (task.id, task.setting, task.sensitivity, metadata.HKParams.Tolerance, metadata.PostRetainCount,
            metadata.HKParams.Influence, metadata.HKParams.RewiringRate, metadata.HKParams.RepostRate,
            task.replicate)


def _pathway_values(point):
    return tuple(getattr(point.pathway, name) for name in PATHWAY_COLUMNS)


def reference_tables(plan, records):
    """Project typed reference and checkpoint observations into fixed table columns."""
    references, selections = [], []
    if plan.profile != "staged":
        return pd.DataFrame(), pd.DataFrame()
    for task in plan.tasks:
        if task.id not in records:
            continue
        record = records[task.id]
        outcome = shared.outcome(record.reference)
        references.extend(ReferenceSample(task, outcome, point) for point in record.reference.endpoints)
        selections.extend(SelectionSample(task, outcome, stage, block.checkpoint) for stage, block in record.stages.items())
    exposure_columns = (*shared.EXPOSURE_COUNTS,
                        *(f"{source}_{name}" for source in shared.SLOT_SOURCES for name in shared.SLOT_MEASURES))
    reference_rows = [
        (*_coordinate_values(row.task), row.reference_outcome, row.point.step, *_pathway_values(row.point),
         *(getattr(row.point.exposure, name) for name in shared.EXPOSURE_COUNTS),
         *(getattr(getattr(row.point.exposure, source), name)
           for source in shared.SLOT_SOURCES for name in shared.SLOT_MEASURES))
        for row in references
    ]
    selection_rows = [
        (*_coordinate_values(row.task), row.reference_outcome, row.stage, row.point.step, *_pathway_values(row.point),
         shared._ratio(row.point.exposure.absolute_shift, row.point.exposure.updates),
         shared._ratio(row.point.exposure.rewires, row.point.exposure.updates))
        for row in selections
    ]
    return (
        pd.DataFrame.from_records(reference_rows, columns=(*COORDINATE_COLUMNS, "reference_outcome", "step",
                                                          *PATHWAY_COLUMNS, *exposure_columns)),
        pd.DataFrame.from_records(selection_rows, columns=(*COORDINATE_COLUMNS, "reference_outcome", "stage", "step",
                                                          *PATHWAY_COLUMNS, "opinion_shift_per_update", "rewires_per_update")),
    )


def stage_interactions(runs):
    rows = []
    for identifying, block in shared.analysis_strata(runs):
        treated = block.loc[block["arm"] == "switch"].set_index(["task_id", "stage"])
        control = block.loc[block["arm"] == "hold"].set_index(["task_id", "stage"])
        for metric in shared.EFFECTS:
            effects = (treated[metric].astype(float) - control[metric].astype(float)).unstack("stage")
            for earlier, later in combinations(cfg.STAGES, 2):
                values = effects[later] - effects[earlier] if {earlier, later} <= set(effects.columns) else []
                estimate = shared.mean_interval(values)
                rows.append(StageInteraction(
                    sensitivity=identifying.sensitivity, epsilon=identifying.epsilon, history=identifying.history,
                    setting=identifying.setting, repost=identifying.repost, earlier=earlier, later=later,
                    metric=metric, n=estimate.n, mean=estimate.mean, lo=estimate.lo, hi=estimate.hi,
                ))
    columns = ("sensitivity", "epsilon", "history", "setting", "repost", "earlier", "later", "metric", "n", "mean", "lo", "hi")
    return rows_frame(rows, StageInteraction).reindex(columns=columns)
