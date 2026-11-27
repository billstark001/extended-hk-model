"""Recorded metrics, paired statistics and table helpers shared by recommendation studies."""

from importlib import import_module

from itertools import chain
from pathlib import Path
from dataclasses import fields

import numpy as np
import pandas as pd
from scipy.stats import t

from ehk.domain import (AnalysisStratum, CoverageRow, MeanInterval, FullResult, StagedResult,
                        read_plan, rows_frame)
from .storage import StudyStore
from .analysis_types import ExposureColumns, ArmSummary, RunIdentity, StudyRunRow, TrajectoryRow, ContrastRow
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
        plan.study != "paper_recommendation"
        or (study is not None and study != plan.profile)
        or (profiles is not None and plan.profile not in profiles)
    ):
        raise ValueError(f"unexpected experiment plan at {directory}")
    modules = {
        "staged": "experiments.paper.recommendation.design",
        "staged_timing": "experiments.paper.recommendation.design",
        "recommender_variants": "experiments.paper.recommender_variants.design",
    }
    if plan.profile not in modules:
        raise ValueError(f"unexpected experiment profile: {plan.profile}")
    design = import_module(modules[plan.profile])
    if plan.schema != design.DESIGN_VERSION:
        raise ValueError("unsupported recommendation design version")

    records = {}
    with StudyStore(directory, readonly=True) as store:
        for task in plan.tasks:
            entry = store.get_task(task, plan)
            if entry is not None:
                records[task.id] = entry.result
    return plan, records


def _arm_binary_sha256(binding, name, arm):
    expected = None
    if binding is not None:
        expected = binding.arm_binary_sha256[name] if binding.arm_binary_sha256 is not None else binding.binary_sha256
    recorded = arm.binary_sha256 if arm is not None else None
    if recorded is not None and expected is not None and recorded != expected:
        raise ValueError(f"arm binary does not match its frozen binding: {name}")
    return recorded or expected


def _ratio(n, d):
    return n / d if d and n is not None else None


def outcome(arm):
    terminal = arm.terminal
    return "unresolved" if terminal.status != "absorbed" else OUTCOMES[min(terminal.k_major, 3)-1]


def exposure_columns(exposure):
    return ExposureColumns(
        updates=exposure.updates,
        requested=exposure.requested,
        neighbor_seen=exposure.neighbor_seen,
        neighbor_concordant=exposure.neighbor_concordant,
        neighbor_with_posts=exposure.neighbor_with_posts,
        neighbor_share_sum=exposure.neighbor_share_sum,
        recommended=exposure.recommended,
        cross_camp=exposure.cross_camp,
        cross_confident=exposure.cross_confident,
        cross_rewire=exposure.cross_rewire,
        rewires=exposure.rewires,
        absolute_shift=exposure.absolute_shift,
        exploration_shown=exposure.exploration.shown,
        exploration_cross_camp=exposure.exploration.cross_camp,
        exploration_cross_confident=exposure.exploration.cross_confident,
        exploration_cross_rewire=exposure.exploration.cross_rewire,
        policy_shown=exposure.policy.shown,
        policy_cross_camp=exposure.policy.cross_camp,
        policy_cross_confident=exposure.policy.cross_confident,
        policy_cross_rewire=exposure.policy.cross_rewire,
        bridge_shown=exposure.bridge.shown,
        bridge_cross_camp=exposure.bridge.cross_camp,
        bridge_cross_confident=exposure.bridge.cross_confident,
        bridge_cross_rewire=exposure.bridge.cross_rewire,
    )


def exposure_values(exposure):
    return tuple(getattr(exposure, item.name) for item in fields(ExposureColumns))


EXPOSURE_COLUMN_NAMES = tuple(item.name for item in fields(ExposureColumns))


def _passage(points):
    return first_passage((point.step, point.pathway.separation, point.pathway.excess_concordance) for point in points)


def summarize_arm(arm, prefix=()):
    points = arm.endpoints
    initial, final = points[0].pathway, points[-1].pathway
    windows = [exposure_columns(point.exposure) for point in points]
    totals = ExposureColumns(**{item.name: sum(getattr(row, item.name) for row in windows) for item in fields(ExposureColumns)})
    n = totals.updates
    passage, window = _passage(chain(prefix, points)), _passage(points)
    separation_crossed = initial.separation >= SEPARATION_THRESHOLD
    segregation_crossed = initial.excess_concordance is not None and initial.excess_concordance >= EXCESS_THRESHOLD
    peaks = opinion_peak_count(np.asarray(arm.final_opinions))
    excess0, excess1 = initial.excess_concordance, final.excess_concordance
    return ArmSummary(
        start_step=points[0].step, end_step=points[-1].step,
        switch_step=arm.switch_step, switch_fallback=arm.switch_fallback,
        raw_record=arm.raw_record, exposure_trace=arm.exposure_trace.path if arm.exposure_trace else None,
        start_separation=initial.separation, end_separation=final.separation,
        separation_change=final.separation-initial.separation,
        start_excess=excess0, end_excess=excess1,
        excess_change=excess1-excess0 if excess0 is not None and excess1 is not None else None,
        end_edge_concordance=final.edge_concordance, end_pair_concordance=final.pair_concordance,
        outcome=outcome(arm), opinion_peaks=peaks, opinion_shape=PEAK_GROUPS[min(peaks,3)-1],
        pathway=passage.label, separation_step=passage.separation_step, segregation_step=passage.segregation_step,
        window_pathway="left_censored" if separation_crossed or segregation_crossed else window.label,
        window_separation_step=None if separation_crossed else window.separation_step,
        window_segregation_step=None if segregation_crossed else window.segregation_step,
        start_separation_crossed=separation_crossed, start_segregation_crossed=segregation_crossed,
        fill=_ratio(totals.recommended,totals.requested), cross_camp_per_update=_ratio(totals.cross_camp,n),
        cross_share=_ratio(totals.cross_camp,totals.recommended),
        followee_post_concordance=_ratio(totals.neighbor_share_sum,totals.neighbor_with_posts),
        rewires_per_update=_ratio(totals.rewires,n), opinion_shift_per_update=_ratio(totals.absolute_shift,n),
        updates=totals.updates,
        requested=totals.requested,
        neighbor_seen=totals.neighbor_seen,
        neighbor_concordant=totals.neighbor_concordant,
        neighbor_with_posts=totals.neighbor_with_posts,
        neighbor_share_sum=totals.neighbor_share_sum,
        recommended=totals.recommended,
        cross_camp=totals.cross_camp,
        cross_confident=totals.cross_confident,
        cross_rewire=totals.cross_rewire,
        rewires=totals.rewires,
        absolute_shift=totals.absolute_shift,
        exploration_shown=totals.exploration_shown,
        exploration_cross_camp=totals.exploration_cross_camp,
        exploration_cross_confident=totals.exploration_cross_confident,
        exploration_cross_rewire=totals.exploration_cross_rewire,
        policy_shown=totals.policy_shown,
        policy_cross_camp=totals.policy_cross_camp,
        policy_cross_confident=totals.policy_cross_confident,
        policy_cross_rewire=totals.policy_cross_rewire,
        bridge_shown=totals.bridge_shown,
        bridge_cross_camp=totals.bridge_cross_camp,
        bridge_cross_confident=totals.bridge_cross_confident,
        bridge_cross_rewire=totals.bridge_cross_rewire,
        non_exploration_shown=totals.policy_shown+totals.bridge_shown,
        non_exploration_cross_camp=totals.policy_cross_camp+totals.bridge_cross_camp,
        non_exploration_cross_confident=totals.policy_cross_confident+totals.bridge_cross_confident,
        non_exploration_cross_rewire=totals.policy_cross_rewire+totals.bridge_cross_rewire,
    )


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return MeanInterval(n=0, mean=None, lo=None, hi=None)
    mean = float(values.mean())
    if n == 1:
        return MeanInterval(n=1, mean=mean, lo=None, hi=None)
    half = float(t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))
    return MeanInterval(n=n, mean=mean, lo=mean-half, hi=mean+half)


def analysis_strata(runs):
    for (sensitivity, epsilon, history), block in runs.groupby(["sensitivity", "epsilon", "history"]):
        yield AnalysisStratum(sensitivity=sensitivity, epsilon=epsilon, history=history, setting="all", repost="all"), block
        for (setting, repost), sample in block.groupby(["setting", "repost"]):
            yield AnalysisStratum(sensitivity=sensitivity, epsilon=epsilon, history=history, setting=setting, repost=repost), sample


def _identity_values(identity, *, staged):
    names = [item.name for item in fields(RunIdentity) if item.name != "reference_steps"]
    if staged:
        names.insert(9, "reference_steps")
    return tuple(getattr(identity, name) for name in names), names


def study_tables(plan, records, *, binding=None, arm_columns=None):
    rows, trajectories = [], []
    staged = plan.profile == "staged"
    for task in plan.tasks:
        if task.id not in records:
            continue
        record, metadata = records[task.id], task.metadata
        blocks = record.stages.items() if isinstance(record, StagedResult) else (("full", record),)
        for stage, block in blocks:
            for name, arm in block.arms.items():
                identity = RunIdentity(task_id=task.id, setting=task.setting, sensitivity=task.sensitivity,
                    epsilon=metadata.HKParams.Tolerance, history=metadata.PostRetainCount,
                    influence=metadata.HKParams.Influence, rewiring=metadata.HKParams.RewiringRate,
                    replicate=task.replicate, repost=metadata.HKParams.RepostRate,
                    stage=stage, arm=name, binary_sha256=_arm_binary_sha256(binding,name,arm),
                    reference_steps=record.reference.stopped_at if staged else None)
                prefix = (point for point in record.reference.endpoints if point.step < arm.endpoints[0].step) if staged else ()
                factors = arm_columns(metadata,name) if arm_columns else None
                rows.append(StudyRunRow(identity=identity, summary=summarize_arm(arm,prefix), factors=factors))
                trajectories.extend(TrajectoryRow(identity=identity, step=point.step, relative_step=point.relative_step,
                    pathway=point.pathway, exposure=exposure_columns(point.exposure)) for point in arm.endpoints)
    factor_names = ("recommender", "zeta", "score_noise") if arm_columns else ()
    identity_names = [item.name for item in fields(RunIdentity) if item.name != "reference_steps"]
    if staged:
        identity_names.insert(9,"reference_steps")
    summary_names = [item.name for item in fields(ArmSummary)]
    run_columns = [*identity_names,*factor_names,*summary_names]
    run_values = [(*_identity_values(row.identity,staged=staged)[0],
                   *(getattr(row.factors,name) for name in factor_names),
                   *(getattr(row.summary,name) for name in summary_names)) for row in rows]
    pathway_names = ("separation","pair_concordance","edge_concordance","excess_concordance")
    trajectory_values = [(*_identity_values(row.identity,staged=staged)[0],row.step,row.relative_step,
                          *(getattr(row.pathway,name) for name in pathway_names),*exposure_values(row.exposure))
                         for row in trajectories]
    return (pd.DataFrame.from_records(run_values,columns=run_columns),
            pd.DataFrame.from_records(trajectory_values,columns=[*identity_names,"step","relative_step",*pathway_names,*EXPOSURE_COLUMN_NAMES]))


def coverage_by_setting(plan, records, *, completed=None, stages=()):
    groups = {}
    completed = set(records) if completed is None else completed
    for task in plan.tasks:
        metadata = task.metadata
        repost = metadata.HKParams.RepostRate
        key = task.setting, repost
        if key not in groups:
            groups[key] = CoverageRow(setting=task.setting, sensitivity=task.sensitivity,
                epsilon=metadata.HKParams.Tolerance, history=metadata.PostRetainCount,
                influence=metadata.HKParams.Influence, rewiring=metadata.HKParams.RewiringRate,
                repost=repost, planned=0, completed=0,
                stages={stage: 0 for stage in stages} if stages else None)
        row = groups[key]
        row.planned += 1
        if task.id in completed:
            row.completed += 1
            if stages:
                for stage in records[task.id].stages:
                    row.stages[stage] += 1
    return list(groups.values())


def completed_cell_task_ids(plan, records, *, coverage=None):
    cells = coverage if coverage is not None else coverage_by_setting(plan, records)
    complete = {(cell.setting, cell.repost) for cell in cells if cell.completed == cell.planned}
    return {task.id for task in plan.tasks
            if task.id in records and (task.setting, task.metadata.HKParams.RepostRate) in complete}


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
                    deltas = treated.loc[ids,metric].astype(float)-baseline.loc[ids,metric].astype(float)
                    interval = mean_interval(deltas)
                    summaries.append(ContrastRow(sensitivity=identifying.sensitivity,epsilon=identifying.epsilon,
                        history=identifying.history,setting=identifying.setting,repost=identifying.repost,
                        stage=stage,arm=treatment,control=control,metric=metric,
                        n=interval.n,mean=interval.mean,lo=interval.lo,hi=interval.hi))
    return rows_frame(summaries,ContrastRow)


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
