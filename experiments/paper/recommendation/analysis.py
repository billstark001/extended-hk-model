"""Analyze matched checkpoint and online-timing interventions."""

from itertools import combinations

import pandas as pd

from experiments.paper.recommendation_shared import analysis as shared

from . import design as cfg


def read_study(directory, study=None):
    return shared.read_study(directory, study, profiles=cfg.STUDIES)


def completed_task_ids(plan, records):
    if plan["profile"] not in cfg.STUDIES:
        raise ValueError(f"not an intervention profile: {plan['profile']}")
    return {task["id"] for task in plan["tasks"] if task["id"] in records}


def study_tables(plan, records, *, binding=None):
    if plan["profile"] not in cfg.STUDIES:
        raise ValueError(f"not an intervention profile: {plan['profile']}")
    return shared.study_tables(plan, records, binding=binding)


def coverage_by_setting(plan, records):
    return shared.coverage_by_setting(
        plan, records, completed=completed_task_ids(plan, records),
        stages=cfg.STAGES if plan["profile"] == "staged" else (),
    )


def completed_cell_task_ids(plan, records):
    return shared.completed_cell_task_ids(plan, records, coverage=coverage_by_setting(plan, records))


def contrast_table(runs, study):
    pairs = {
        "staged": [("switch", "hold"), ("bridge", "switch"), ("hide_reposts", "switch")],
        "staged_timing": [("state", "fixed"), ("state", "hold"), ("state", "always_structure")],
    }[study]
    return shared.contrast_table(runs, pairs)


def stage_gate(tasks, records):
    core = [task for task in tasks if task["sensitivity"] == "core"]
    differences = []
    for task in core:
        if task["id"] not in records:
            continue
        record = records[task["id"]]
        phases = record["stages"]
        if "early" in phases and "late" in phases:
            effects = []
            for phase in ("early", "late"):
                arms = phases[phase]["arms"]
                effects.append(
                    arms["switch"]["endpoints"][-1]["pathway"]["separation"]
                    - arms["hold"]["endpoints"][-1]["pathway"]["separation"]
                )
            differences.append(effects[1] - effects[0])
    stats = shared.mean_interval(differences)
    lower, upper = stats["lo"], stats["hi"]
    completed = sum(task["id"] in records for task in core)
    complete = bool(core) and completed == len(core)
    passed = bool(
        complete
        and lower is not None
        and upper is not None
        and abs(stats["mean"]) >= cfg.MIN_STAGE_EFFECT
        and (lower > 0 or upper < 0)
    )
    return {
        **stats, "planned_core": len(core), "completed_core": completed,
        "complete": complete, "passed": passed, "minimum_effect": cfg.MIN_STAGE_EFFECT,
    }


def reference_tables(plan, records):
    """Expose full reference histories and activity-selected intervention states."""
    references, selections = [], []
    if plan["profile"] != "staged":
        return pd.DataFrame(), pd.DataFrame()
    for task in plan["tasks"]:
        if task["id"] not in records:
            continue
        metadata = task["metadata"]
        ids = {
            "task_id": task["id"], "setting": task["setting"], "sensitivity": task["sensitivity"],
            "epsilon": metadata["HKParams"]["Tolerance"], "history": metadata["PostRetainCount"],
            "influence": metadata["HKParams"]["Influence"], "rewiring": metadata["HKParams"]["RewiringRate"],
            "repost": metadata["HKParams"]["RepostRate"], "replicate": task["replicate"],
        }
        record = records[task["id"]]
        ids["reference_outcome"] = shared.outcome(record["reference"])
        for point in record["reference"]["endpoints"]:
            references.append({**ids, "step": point["step"], **point["pathway"], **shared.exposure_columns(point["exposure"])})
        for stage, block in record["stages"].items():
            point = block["checkpoint"]
            exposure = point["exposure"]
            selections.append({
                **ids, "stage": stage, "step": point["step"], **point["pathway"],
                "opinion_shift_per_update": shared._ratio(exposure["absolute_shift"], exposure["updates"]),
                "rewires_per_update": shared._ratio(exposure["rewires"], exposure["updates"]),
            })
    return pd.DataFrame(references), pd.DataFrame(selections)


def stage_interactions(runs):
    """Compare switch effects on the same reference trajectories across stages."""
    rows = []
    for identifying, block in shared.analysis_strata(runs):
        treated = block.loc[block["arm"] == "switch"].set_index(["task_id", "stage"])
        control = block.loc[block["arm"] == "hold"].set_index(["task_id", "stage"])
        for metric in shared.EFFECTS:
            effects = (treated[metric].astype(float) - control[metric].astype(float)).unstack("stage")
            for earlier, later in combinations(cfg.STAGES, 2):
                values = effects[later] - effects[earlier] if {earlier, later} <= set(effects.columns) else []
                rows.append({**identifying, "earlier": earlier, "later": later, "metric": metric, **shared.mean_interval(values)})
    return pd.DataFrame(rows)
