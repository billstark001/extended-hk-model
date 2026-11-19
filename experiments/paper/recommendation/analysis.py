"""Summarize completed paired experiments on their recorded observation grid."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from common.experiments.plans import read_plan
from common.io.json import fingerprint, read_result
from ehk.metrics.pathway import first_passage
from ehk.metrics.polarization import opinion_peak_count

from . import design as cfg

OUTCOMES = ("consensus", "two groups", "three or more groups", "unresolved")
PEAK_GROUPS = ("one mode", "two modes", "three or more modes")
EFFECTS = (
    "separation_change",
    "excess_change",
    "cross_camp_per_update",
    "fill",
    "rewires_per_update",
    "opinion_shift_per_update",
)


def read_study(directory: Path, study: str | None = None):
    plan = read_plan(directory / "plan.json")
    if plan["study"] != "paper_recommendation" or (study is not None and study != plan["profile"]):
        raise ValueError(f"unexpected experiment plan at {directory}")
    if plan["schema"] != cfg.DESIGN_VERSION:
        raise ValueError("unsupported recommendation design version")

    records = {}
    for task in plan["tasks"]:
        path = directory / "records" / (task["id"] + ".json.gz")
        if not path.is_file():
            continue
        entry = read_result(path)
        if entry["task_fingerprint"] != fingerprint(task):
            raise ValueError(f"record does not match design: {path}")
        records[task["id"]] = entry["result"]
    return plan, records


def _ratio(n, d):
    return n / d if d and n is not None else None


def outcome(arm):
    terminal = arm["terminal"]
    if terminal["status"] != "absorbed":
        return "unresolved"
    return OUTCOMES[min(terminal["k_major"], 3) - 1]


SLOT_SOURCES = ("exploration", "policy", "bridge")
SLOT_MEASURES = ("shown", "cross_camp", "cross_confident", "cross_rewire")
EXPOSURE_COUNTS = (
    "updates",
    "requested",
    "neighbor_seen",
    "neighbor_concordant",
    "neighbor_with_posts",
    "neighbor_share_sum",
    "recommended",
    "cross_camp",
    "cross_confident",
    "cross_rewire",
    "rewires",
    "absolute_shift",
)


def exposure_columns(exposure: dict) -> dict:
    values = {key: exposure[key] for key in EXPOSURE_COUNTS}
    for source in SLOT_SOURCES:
        values.update({f"{source}_{key}": exposure[source][key] for key in SLOT_MEASURES})
    return values


def summarize_arm(arm):
    points = arm["endpoints"]
    initial, final = (point["pathway"] for point in (points[0], points[-1]))
    windows = [exposure_columns(point["exposure"]) for point in points]
    totals = {
        key: sum(row[key] for row in windows) if all(row[key] is not None for row in windows) else None
        for key in windows[0]
    }
    n = totals["updates"]
    passage = first_passage(
        (point["step"], point["pathway"]["separation"], point["pathway"]["excess_concordance"]) for point in points
    )
    peaks = opinion_peak_count(np.asarray(arm["final_opinions"]))
    excess0, excess1 = initial["excess_concordance"], final["excess_concordance"]
    return {
        "start_step": points[0]["step"],
        "end_step": points[-1]["step"],
        "switch_step": arm.get("switch_step"),
        "switch_fallback": arm.get("switch_fallback", False),
        "raw_record": arm["raw_record"],
        "exposure_trace": arm.get("exposure_trace", {}).get("path"),
        "start_separation": initial["separation"],
        "end_separation": final["separation"],
        "separation_change": final["separation"] - initial["separation"],
        "start_excess": excess0,
        "end_excess": excess1,
        "excess_change": excess1 - excess0 if excess0 is not None and excess1 is not None else None,
        "end_edge_concordance": final["edge_concordance"],
        "end_pair_concordance": final["pair_concordance"],
        "outcome": outcome(arm),
        "opinion_peaks": peaks,
        "opinion_shape": PEAK_GROUPS[min(peaks, 3) - 1],
        "pathway": passage.label,
        "separation_step": passage.separation_step,
        "segregation_step": passage.segregation_step,
        "fill": _ratio(totals["recommended"], totals["requested"]),
        "cross_camp_per_update": _ratio(totals["cross_camp"], n),
        "cross_share": _ratio(totals["cross_camp"], totals["recommended"]),
        "followee_post_concordance": _ratio(totals["neighbor_share_sum"], totals["neighbor_with_posts"]),
        "rewires_per_update": _ratio(totals["rewires"], n),
        "opinion_shift_per_update": _ratio(totals["absolute_shift"], n),
        **totals,
    }


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    mean = float(values.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "lo": None, "hi": None}
    half = float(t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))
    return {"n": n, "mean": mean, "lo": mean - half, "hi": mean + half}


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
    stats = mean_interval(differences)
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


def study_tables(plan, records):
    rows, trajectories = [], []
    for task in plan["tasks"]:
        if task["id"] not in records:
            continue
        record = records[task["id"]]
        metadata = task["metadata"]
        identifying = {
            "task_id": task["id"],
            "setting": task["setting"],
            "sensitivity": task["sensitivity"],
            "epsilon": metadata["HKParams"]["Tolerance"],
            "history": metadata["PostRetainCount"],
            "influence": metadata["HKParams"]["Influence"],
            "rewiring": metadata["HKParams"]["RewiringRate"],
            "replicate": task["replicate"],
            "repost": metadata["HKParams"]["RepostRate"],
        }
        stages = (
            {stage: block["arms"] for stage, block in record["stages"].items()}
            if plan["profile"] == "staged"
            else {"full": record["arms"]}
        )
        if plan["profile"] == "staged":
            identifying["reference_steps"] = record["reference"]["stopped_at"]
        for stage, arms in stages.items():
            for name, arm in arms.items():
                ids = {**identifying, "stage": stage, "arm": name}
                rows.append({**ids, **summarize_arm(arm)})
                for point in arm["endpoints"]:
                    trajectories.append(
                        {
                            **ids,
                            "step": point["step"],
                            "relative_step": point["relative_step"],
                            **point["pathway"],
                            **exposure_columns(point["exposure"]),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(trajectories)


def coverage_by_setting(plan, records):
    groups = {}
    for task in plan["tasks"]:
        metadata = task["metadata"]
        repost = metadata["HKParams"]["RepostRate"]
        key = task["setting"], repost
        if key not in groups:
            groups[key] = {
                "setting": task["setting"], "sensitivity": task["sensitivity"],
                "epsilon": metadata["HKParams"]["Tolerance"], "history": metadata["PostRetainCount"],
                "influence": metadata["HKParams"]["Influence"], "rewiring": metadata["HKParams"]["RewiringRate"],
                "repost": repost, "planned": 0, "completed": 0,
            }
            if plan["profile"] == "staged":
                groups[key]["stages"] = {stage: 0 for stage in cfg.STAGES}
        row = groups[key]
        row["planned"] += 1
        if task["id"] in records:
            row["completed"] += 1
            if plan["profile"] == "staged":
                for stage in records[task["id"]]["stages"]:
                    row["stages"][stage] += 1
    return list(groups.values())


def analysis_strata(runs):
    """Pool alpha/q/repost cells only within a fixed epsilon and history."""
    for (sensitivity, epsilon, history), block in runs.groupby(["sensitivity", "epsilon", "history"]):
        common = {"sensitivity": sensitivity, "epsilon": epsilon, "history": history}
        yield {**common, "setting": "all", "repost": "all"}, block
        for (setting, repost), sample in block.groupby(["setting", "repost"]):
            yield {**common, "setting": setting, "repost": repost}, sample


def contrast_table(runs, study):
    pairs = {
        "staged": [("switch", "hold"), ("bridge", "switch"), ("hide_reposts", "switch")],
        "staged_timing": [("state", "fixed"), ("state", "hold"), ("state", "always_structure")],
        "recommender_variants": [
            (policy, family[1]) for family in cfg.VARIANTS.values() for policy in (family[0], family[2])
        ]
        + list(zip(cfg.VARIANTS["Opinion"], cfg.VARIANTS["Structure"])),
    }[study]
    summaries = []
    for identifying, block in analysis_strata(runs):
        for stage, phases in block.groupby("stage"):
            for treatment, control in pairs:
                treated = phases.loc[phases["arm"] == treatment].set_index("task_id")
                baseline = phases.loc[phases["arm"] == control].set_index("task_id")
                ids = treated.index.intersection(baseline.index)
                if not len(ids):
                    continue
                for metric in EFFECTS:
                    deltas = treated.loc[ids, metric].astype(float) - baseline.loc[ids, metric].astype(float)
                    summaries.append(
                        {
                            **identifying,
                            "stage": stage,
                            "arm": treatment,
                            "control": control,
                            "metric": metric,
                            **mean_interval(deltas),
                        }
                    )
    return pd.DataFrame(summaries)


def stage_interactions(runs):
    """Compare switch effects on the same reference trajectories across stages."""
    rows = []
    for identifying, block in analysis_strata(runs):
        treated = block.loc[block["arm"] == "switch"].set_index(["task_id", "stage"])
        control = block.loc[block["arm"] == "hold"].set_index(["task_id", "stage"])
        for metric in EFFECTS:
            effects = (treated[metric].astype(float) - control[metric].astype(float)).unstack("stage")
            for earlier, later in combinations(cfg.STAGES, 2):
                values = effects[later] - effects[earlier] if {earlier, later} <= set(effects.columns) else []
                rows.append({**identifying, "earlier": earlier, "later": later, "metric": metric, **mean_interval(values)})
    return pd.DataFrame(rows)
