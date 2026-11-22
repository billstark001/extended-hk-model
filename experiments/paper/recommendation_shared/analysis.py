"""Recorded metrics, paired statistics and table helpers shared by recommendation studies."""

from importlib import import_module
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from common.experiments.plans import read_plan
from common.io.json import fingerprint, read_result
from ehk.metrics.pathway import EXCESS_THRESHOLD, SEPARATION_THRESHOLD, first_passage
from ehk.metrics.polarization import opinion_peak_count

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


def read_study(directory: Path, study: str | None = None, *, profiles=None):
    plan = read_plan(directory / "plan.json")
    if (
        plan["study"] != "paper_recommendation"
        or (study is not None and study != plan["profile"])
        or (profiles is not None and plan["profile"] not in profiles)
    ):
        raise ValueError(f"unexpected experiment plan at {directory}")
    modules = {
        "staged": "experiments.paper.recommendation.design",
        "staged_timing": "experiments.paper.recommendation.design",
        "recommender_variants": "experiments.paper.recommender_variants.design",
    }
    if plan["profile"] not in modules:
        raise ValueError(f"unexpected experiment profile: {plan['profile']}")
    design = import_module(modules[plan["profile"]])
    if plan["schema"] != design.DESIGN_VERSION:
        raise ValueError("unsupported recommendation design version")

    paths = [(task, directory / "records" / (task["id"] + ".json.gz")) for task in plan["tasks"]]
    available = [(task, path) for task, path in paths if path.is_file()]
    records = {}
    for task, path in available:
        entry = read_result(path)
        if entry["task_fingerprint"] != fingerprint(task):
            raise ValueError(f"record does not match design: {path}")
        records[task["id"]] = entry["result"]
    return plan, records


def _arm_binary_sha256(binding, name, arm):
    expected = None
    if binding is not None:
        expected = binding["arm_binary_sha256"][name] if "arm_binary_sha256" in binding else binding["binary_sha256"]
    recorded = arm.get("binary_sha256")
    if recorded is not None and expected is not None and recorded != expected:
        raise ValueError(f"arm binary does not match its frozen binding: {name}")
    return recorded or expected


def _ratio(n, d):
    return n / d if d and n is not None else None


def outcome(arm):
    terminal = arm["terminal"]
    if terminal["status"] != "absorbed":
        return "unresolved"
    return OUTCOMES[min(terminal["k_major"], 3) - 1]


def exposure_columns(exposure: dict) -> dict:
    values = {key: exposure[key] for key in EXPOSURE_COUNTS}
    for source in SLOT_SOURCES:
        values.update({f"{source}_{key}": exposure[source][key] for key in SLOT_MEASURES})
    return values


def _passage(points):
    return first_passage(
        (point["step"], point["pathway"]["separation"], point["pathway"]["excess_concordance"])
        for point in points
    )


def summarize_arm(arm, prefix=()):
    points = arm["endpoints"]
    initial, final = (point["pathway"] for point in (points[0], points[-1]))
    windows = [exposure_columns(point["exposure"]) for point in points]
    totals = {
        key: sum(row[key] for row in windows) if all(row[key] is not None for row in windows) else None
        for key in windows[0]
    }
    n = totals["updates"]
    passage = _passage(chain(prefix, points))
    window = _passage(points)
    separation_crossed = initial["separation"] >= SEPARATION_THRESHOLD
    segregation_crossed = initial["excess_concordance"] is not None and initial["excess_concordance"] >= EXCESS_THRESHOLD
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
        "window_pathway": "left_censored" if separation_crossed or segregation_crossed else window.label,
        "window_separation_step": None if separation_crossed else window.separation_step,
        "window_segregation_step": None if segregation_crossed else window.segregation_step,
        "start_separation_crossed": separation_crossed,
        "start_segregation_crossed": segregation_crossed,
        "fill": _ratio(totals["recommended"], totals["requested"]),
        "cross_camp_per_update": _ratio(totals["cross_camp"], n),
        "cross_share": _ratio(totals["cross_camp"], totals["recommended"]),
        "followee_post_concordance": _ratio(totals["neighbor_share_sum"], totals["neighbor_with_posts"]),
        "rewires_per_update": _ratio(totals["rewires"], n),
        "opinion_shift_per_update": _ratio(totals["absolute_shift"], n),
        **totals,
        **{f"non_exploration_{key}": totals[f"policy_{key}"] + totals[f"bridge_{key}"] for key in SLOT_MEASURES},
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


def analysis_strata(runs):
    """Pool alpha/q/repost cells only within a fixed epsilon and history."""
    for (sensitivity, epsilon, history), block in runs.groupby(["sensitivity", "epsilon", "history"]):
        common = {"sensitivity": sensitivity, "epsilon": epsilon, "history": history}
        yield {**common, "setting": "all", "repost": "all"}, block
        for (setting, repost), sample in block.groupby(["setting", "repost"]):
            yield {**common, "setting": setting, "repost": repost}, sample


def study_tables(plan, records, *, binding=None, arm_columns=None):
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
                ids = {
                    **identifying, "stage": stage, "arm": name,
                    "binary_sha256": _arm_binary_sha256(binding, name, arm),
                }
                if arm_columns is not None:
                    ids.update(arm_columns(metadata, name))
                start = arm["endpoints"][0]["step"]
                prefix = (
                    (point for point in record["reference"]["endpoints"] if point["step"] < start)
                    if plan["profile"] == "staged" else ()
                )
                rows.append({**ids, **summarize_arm(arm, prefix)})
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


def coverage_by_setting(plan, records, *, completed=None, stages=()):
    groups = {}
    if completed is None:
        completed = set(records)
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
            if stages:
                groups[key]["stages"] = {stage: 0 for stage in stages}
        row = groups[key]
        row["planned"] += 1
        if task["id"] in completed:
            row["completed"] += 1
            if stages:
                for stage in records[task["id"]]["stages"]:
                    row["stages"][stage] += 1
    return list(groups.values())


def completed_cell_task_ids(plan, records, *, coverage=None):
    """Select observed tasks only from parameter cells with every planned repeat."""
    complete = {
        (cell["setting"], cell["repost"])
        for cell in (coverage if coverage is not None else coverage_by_setting(plan, records)) if cell["completed"] == cell["planned"]
    }
    return {
        task["id"] for task in plan["tasks"]
        if task["id"] in records and (task["setting"], task["metadata"]["HKParams"]["RepostRate"]) in complete
    }


def contrast_table(runs, pairs):
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
    return pd.DataFrame(summaries, columns=[
        "sensitivity", "epsilon", "history", "setting", "repost", "stage", "arm", "control", "metric",
        "n", "mean", "lo", "hi",
    ])


def paired_effects(runs, arm, control):
    """Join within task and checkpoint, rejecting ambiguous or mismatched pairs."""
    selected = runs[runs["arm"].isin((arm, control))]
    keys = ["task_id", "stage"]
    if selected.duplicated([*keys, "arm"]).any():
        raise ValueError("duplicate task/stage/arm observations")
    left, right = (selected[selected["arm"] == name].set_index(keys) for name in (arm, control))
    joined = left.join(right, how="inner", lsuffix="_arm", rsuffix="_control")
    for name in ("start_step", "start_separation", "start_excess"):
        a, b = joined[name + "_arm"], joined[name + "_control"]
        if not ((a == b) | (a.isna() & b.isna())).all():
            raise ValueError(f"paired branches have different {name}")
    result = joined[[name + "_arm" for name in (
        "setting", "sensitivity", "influence", "rewiring", "repost", "replicate", "epsilon", "history",
        "bridge_shown", "cross_confident", "updates",
    )]].rename(columns=lambda name: name.removesuffix("_arm"))
    for name in ("separation", "excess"):
        result["delta_" + name] = joined["end_" + name + "_arm"] - joined["end_" + name + "_control"]
    return result.reset_index()
