"""Audit KDE approximation errors on an authenticated, explicitly selected cohort.

This module reads saved measurements. It neither evaluates a KDE nor simulates a
trajectory; the existing paired-analysis functions retain their original scopes.
"""

import argparse
import inspect
import json
from pathlib import Path
from dataclasses import dataclass, fields
import shutil
from common.experiments.artifacts import ArtifactCache
from common.experiments.plans import freeze_manifest
from common.experiments.lifecycle import session_lock
from ehk.domain import rows_frame, to_wire
from experiments.paper.metric_comparison.core import load_contract
from experiments.paper.metric_comparison.contracts import ReportManifest, ReportInputs
from .contracts import Configuration, CollectionManifest, Evidence, Report

import numpy as np
import pandas as pd

from common.io.json import atomic_json, sha
from experiments.paper.metric_comparison.core import (PATH_KEYS, authenticate_preparation, endpoint_mode,
                                                      validate_endpoint_column, _authenticate_files)
from experiments.paper.metric_comparison.report import STRATA, effect_summaries, intervention_scopes, paired_table
from experiments.paper.recommendation_shared.analysis import mean_interval

from experiments.paper.recommendation_shared.storage import read_table, write_table

METRICS = {"Ip": "I_p", "Is": "I_s", "Ih": "I_h"}
GRIDS = ("prepared", "baseline_adaptive", "candidate_adaptive")
ROUNDING_TOLERANCE = 64 * np.finfo(np.float64).eps
EFFECT_KEYS = [*STRATA, "stage", "contrast", "scope", "metric"]


def _csv(path):
    path = Path(path)
    return read_table(path.parent, path.name)


def _manifest(directory, name, contract):
    directory = Path(directory).resolve()
    result = load_contract(directory / name, contract)
    _authenticate_files(directory, result)
    return result


def _required(frame, columns, label):
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{label} omits required columns: {sorted(missing)}")


def _same(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= ROUNDING_TOLERANCE * np.maximum(1, np.maximum(np.abs(a), np.abs(b))))


def _steps(value, end_step):
    steps = json.loads(value)
    if (not isinstance(steps, list) or not steps or any(type(step) is not int for step in steps)
            or steps != sorted(set(steps)) or steps[0] != 0 or steps[-1] != end_step):
        raise ValueError("adaptive sampling must retain a sorted unique full 0..end prefix")
    return steps


def load_inputs(preparation, baseline_report, collection, *, allow_partial=False):
    """Authenticate generations and exact path/state joins before deriving errors."""
    preparation, baseline_report, collection = (Path(p).resolve() for p in (preparation, baseline_report, collection))
    prepared = authenticate_preparation(preparation)
    baseline = _manifest(baseline_report, "report.json", ReportManifest)
    collected = _manifest(collection, "collection.json", CollectionManifest)
    configuration = load_contract(collection / "configuration.json", Configuration)
    endpoint_complement = endpoint_mode(configuration)
    if endpoint_mode(baseline) != endpoint_complement or endpoint_mode(collected) != endpoint_complement:
        raise ValueError("baseline and approximation endpoint_complement policies disagree")
    if baseline.status != "complete" or baseline.preparation_sha256 != sha(preparation / "preparation.json"):
        raise ValueError("baseline is incomplete or belongs to another preparation")
    if collected.configuration_sha256 != sha(collection / "configuration.json"):
        raise ValueError("collection configuration hash mismatch")
    if configuration.baseline_report.sha256 != sha(baseline_report / "report.json"):
        raise ValueError("collection belongs to another frozen baseline")
    if sha(Path(inspect.getfile(paired_table))) != baseline.report_source_sha256:
        raise ValueError("existing paired-analysis implementation differs from the frozen baseline")
    if collected.status != "complete" and not allow_partial:
        raise ValueError("approximation collection is incomplete")
    points, paths, benchmarks = (_csv(collection / name) for name in ("points.csv.gz", "paths.csv.gz", "benchmarks.csv"))
    paths = validate_endpoint_column(paths, endpoint_complement)
    endpoints = _csv(baseline_report / "endpoint_states.csv.gz")
    endpoints = validate_endpoint_column(endpoints, endpoint_complement)
    _required(paths, [*PATH_KEYS, "bins", "end_step", "baseline_steps_json", "candidate_steps_json",
                      *[f"{mode}_{quantity}" for mode in ("baseline", "fixed", "adaptive")
                        for quantity in ("Iw", "active_step")]], "paths")
    _required(points, [*PATH_KEYS, "bins", "step", "grid",
                       *[f"{method}_{metric}" for method in ("exact", "approx") for metric in METRICS]], "points")
    if points.duplicated([*PATH_KEYS, "bins", "grid", "step"]).any() or paths.duplicated([*PATH_KEYS, "bins"]).any():
        raise ValueError("duplicate approximation path/state identity")
    if paths.empty or set(points["grid"]) - set(GRIDS):
        raise ValueError("empty cohort or unknown measurement grid")
    bins = sorted(paths["bins"].unique().tolist())
    if any(value not in (1021, 2041, 4081, 8161) for value in bins) or set(points["bins"]) != set(bins):
        raise ValueError("unexpected or inconsistent KDE grid sizes")
    if set(configuration.bins) != set(bins):
        raise ValueError("collected KDE grids differ from the frozen configuration")
    if not benchmarks.empty and set(benchmarks["bins"]) != set(bins):
        raise ValueError("benchmark KDE grids differ from the measured state grids")
    selected = paths[PATH_KEYS].drop_duplicates()
    selected_groups = selected[["profile", "task_id"]].drop_duplicates()
    expected = endpoints.merge(selected_groups, on=["profile", "task_id"], validate="many_to_one")
    if set(map(tuple, selected.to_numpy())) != set(map(tuple, expected[PATH_KEYS].to_numpy())):
        raise ValueError("selected task groups omit baseline stages or policies")
    selected_keys = set(map(tuple, selected.to_numpy()))
    selection = configuration.selection
    if (selection.selected_tasks != len(selected_groups)
            or selection.selected_paths != len(selected)):
        raise ValueError("configuration selection counts disagree with collected paths")
    configured = {(row.profile,row.task_id,row.stage,row.arm) for row in configuration.selected_paths}
    if configured != selected_keys:
        raise ValueError("configuration selection identities disagree with collected paths")
    if selection.mode == "full" and len(selected) != len(endpoints):
        raise ValueError("full-cohort configuration omits frozen baseline paths")
    if collected.status == "complete":
        if (collected.planned_paths != len(selected)
                or collected.completed_paths != len(selected)
                or collected.failed_paths or collected.pending_paths):
            raise ValueError("complete collection manifest has inconsistent path coverage")
    for size, frame in paths.groupby("bins"):
        if set(map(tuple, frame[PATH_KEYS].to_numpy())) != selected_keys:
            raise ValueError(f"B={size} does not cover the same selected cohort")
    prepared_points = _csv(preparation / "points.csv.gz")
    requested = prepared_points.merge(selected, on=PATH_KEYS, validate="many_to_one")
    state_keys = [*PATH_KEYS, "step"]
    requested_states = set(map(tuple, requested[state_keys].to_numpy()))
    for size, frame in points[points["grid"] == "prepared"].groupby("bins"):
        if set(map(tuple, frame[state_keys].to_numpy())) != requested_states:
            raise ValueError(f"B={size} prepared observations differ from the frozen cohort")
    if set(points.loc[points["grid"] == "prepared", "bins"]) != set(bins):
        raise ValueError("a KDE grid lacks prepared observations")
    if set(map(tuple, points[PATH_KEYS].drop_duplicates().to_numpy())) != selected_keys:
        raise ValueError("measurement states differ from the selected paths")
    observed_paths = dict(tuple(points.groupby([*PATH_KEYS, "bins"], sort=False)))
    for row in paths.itertuples(index=False):
        observed = observed_paths[(row.profile,row.task_id,row.stage,row.arm,row.bins)]
        for grid, column in (("baseline_adaptive", "baseline_steps_json"), ("candidate_adaptive", "candidate_steps_json")):
            if observed.loc[observed["grid"] == grid, "step"].sort_values().tolist() != _steps(getattr(row,column), row.end_step):
                raise ValueError("adaptive state rows disagree with their accepted step list")
    exact_required = points[points["grid"].isin(("prepared", "baseline_adaptive"))]
    if (not np.isfinite(exact_required[[f"exact_{m}" for m in METRICS]].to_numpy(float)).all()
            or not np.isfinite(points[[f"approx_{m}" for m in METRICS]].to_numpy(float)).all()):
        raise ValueError("required exact/approximate state values are missing or nonfinite")
    for _, same_state in points.groupby([*PATH_KEYS, "step"], sort=False):
        if any(same_state[f"exact_{m}"].dropna().nunique() > 1 for m in METRICS):
            raise ValueError("exact reference changes between repeated observations of one frozen endpoint/state")
    exact_values = points[[f"exact_{m}" for m in METRICS]].to_numpy(float)
    if np.isinf(exact_values).any():
        raise ValueError("available exact state values are nonfinite")
    available_h = points["exact_Ih"].notna()
    if not _same(points.loc[available_h, "exact_Ih"], points.loc[available_h, "approx_Ih"]).all():
        raise ValueError("KDE-only approximation changed homophily")
    path_values = [f"{mode}_{quantity}" for mode in ("baseline", "fixed", "adaptive") for quantity in ("Iw", "active_step")]
    if not np.isfinite(paths[path_values].to_numpy(float)).all():
        raise ValueError("trajectory values are missing or nonfinite")
    for mode in ("baseline", "fixed", "adaptive"):
        active = paths[f"{mode}_active_step"]
        if ((active < 0) | (active > paths["end_step"]) | (active % 1 != 0)).any():
            raise ValueError("activity step lies outside the observed integer-step trajectory")
    for _, frame in paths.groupby(PATH_KEYS, sort=False):
        if frame["baseline_steps_json"].nunique() != 1 or frame["baseline_active_step"].nunique() != 1 or not _same(frame["baseline_Iw"], frame["baseline_Iw"].iloc[0]).all():
            raise ValueError("exact baseline changes between KDE grid sizes")
        for name in ("extra_exact_calls", "extra_exact_seconds"):
            if name in frame and (frame[name].nunique(dropna=False) != 1 or not np.isfinite(frame[name]).all() or (frame[name] < 0).any()):
                raise ValueError("shared exact reconstruction cost differs between KDE grids")
    matched = paths.merge(expected[[*PATH_KEYS, "end_step", "I_w"]], on=[*PATH_KEYS, "end_step"], validate="many_to_one", how="left")
    if not _same(matched["source_Iw"], matched["I_w"]).all():
        raise ValueError("collected source Iw differs from the authenticated formal analysis")
    for endpoint in ("start", "end"):
        joined = points[points["grid"] == "prepared"].merge(
            expected[[*PATH_KEYS, f"{endpoint}_step", *[f"{endpoint}_{m}" for m in METRICS.values()]]],
            left_on=[*PATH_KEYS, "step"], right_on=[*PATH_KEYS, f"{endpoint}_step"], how="inner", validate="many_to_one")
        if len(joined) != len(paths):
            raise ValueError("a selected path lacks an exact prepared endpoint")
        for short, long in METRICS.items():
            if not _same(joined[f"source_{short}"], joined[f"{endpoint}_{long}"]).all():
                raise ValueError(f"source {long} differs from its authenticated endpoint")
    return points, paths, benchmarks, expected, Evidence(prepared,baseline,collected,configuration)


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    n: int
    mean_error: float
    mean_absolute_error: float
    median_absolute_error: float
    p95_absolute_error: float
    maximum_absolute_error: float


def _error_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    absolute = np.abs(values)
    return ErrorSummary(len(values),values.mean() if len(values) else np.nan,
        absolute.mean() if len(values) else np.nan,np.median(absolute) if len(values) else np.nan,
        np.quantile(absolute,.95) if len(values) else np.nan,absolute.max() if len(values) else np.nan)


def point_errors(points):
    rows = []
    for short, metric in METRICS.items():
        frame = points[[*PATH_KEYS, "bins", "step", "grid", f"exact_{short}", f"approx_{short}"]].rename(
            columns={f"exact_{short}": "exact", f"approx_{short}": "approximate"})
        frame["metric"] = metric
        frame["error"] = frame["approximate"] - frame["exact"]
        frame["absolute_error"] = frame["error"].abs()
        rows.append(frame)
    detailed = pd.concat(rows, ignore_index=True)
    summaries = []
    for key, frame in detailed.groupby(["bins", "grid", "metric"], sort=False):
        error = _error_summary(frame["error"])
        summaries.append((*key,len(frame),int(frame["exact"].isna().sum()),*(getattr(error,item.name) for item in fields(ErrorSummary))))
    worst = detailed.dropna(subset=["error"]).sort_values("absolute_error", ascending=False).groupby(
        ["bins", "grid", "metric"], sort=False).head(10)
    return detailed, pd.DataFrame(summaries,columns=["bins","grid","metric","measured_states","unavailable_exact_states",*[item.name for item in fields(ErrorSummary)]]), worst


def trajectory_errors(paths):
    result = paths.copy()
    counts = []
    for row in paths.itertuples(index=False):
        baseline, candidate = (_steps(getattr(row,name), row.end_step) for name in ("baseline_steps_json", "candidate_steps_json"))
        counts.append((len(baseline),len(candidate),len(set(candidate)-set(baseline)),
                       len(set(baseline)-set(candidate)),baseline == candidate))
    result = pd.concat([result.reset_index(drop=True), pd.DataFrame(counts,columns=["baseline_step_count","candidate_step_count","added_step_count","removed_step_count","same_accepted_steps"])], axis=1)
    result["step_count_difference"] = result["candidate_step_count"] - result["baseline_step_count"]
    for quantity in ("Iw", "active_step"):
        result[f"fixed_{quantity}_error"] = result[f"fixed_{quantity}"] - result[f"baseline_{quantity}"]
        result[f"adaptive_{quantity}_error"] = result[f"adaptive_{quantity}"] - result[f"baseline_{quantity}"]
        result[f"grid_{quantity}_change"] = result[f"adaptive_{quantity}"] - result[f"fixed_{quantity}"]
    rows = []
    for size, frame in result.groupby("bins", sort=False):
        for name in ("fixed_Iw_error", "adaptive_Iw_error", "grid_Iw_change", "fixed_active_step_error",
                     "adaptive_active_step_error", "grid_active_step_change", "step_count_difference"):
            error = _error_summary(frame[name])
            rows.append((size,name,*(getattr(error,item.name) for item in fields(ErrorSummary)),
                int(frame["same_accepted_steps"].sum()),int(frame["added_step_count"].sum()),int(frame["removed_step_count"].sum())))
    return result,pd.DataFrame(rows,columns=["bins","quantity",*[item.name for item in fields(ErrorSummary)],
        "same_accepted_step_paths","added_steps","removed_steps"])


def paired_effects(points, endpoints):
    """Reuse existing contrasts and scopes, preserving states and terminal outcomes."""
    endpoints = endpoints.copy()
    exact_points = points[points["grid"] == "prepared"].drop_duplicates([*PATH_KEYS, "step"])
    for endpoint in ("start", "end"):
        exact = exact_points[[*PATH_KEYS, "step", *[f"exact_{name}" for name in METRICS]]].rename(
            columns={"step": f"{endpoint}_step", **{f"exact_{short}": f"{endpoint}_{long}" for short, long in METRICS.items()}})
        endpoints = endpoints.drop(columns=[f"{endpoint}_{long}" for long in METRICS.values()]).merge(
            exact, on=[*PATH_KEYS, f"{endpoint}_step"], how="left", validate="one_to_one")
    baseline_pairs = paired_table(endpoints)
    if baseline_pairs.empty:
        return (pd.DataFrame(columns=[*PATH_KEYS, "bins", "metric", "delta_error"]),
                pd.DataFrame(columns=[*EFFECT_KEYS, "bins", "mean_error"]), baseline_pairs)
    baseline_effects = effect_summaries(baseline_pairs)
    paired_rows, effect_rows = [], []
    for size, frame in points[points["grid"] == "prepared"].groupby("bins", sort=False):
        candidate = endpoints.copy()
        for endpoint in ("start", "end"):
            selected = frame[[*PATH_KEYS, "step", "approx_Ip", "approx_Is"]].rename(
                columns={"step": f"{endpoint}_step"})
            candidate = candidate.merge(selected, on=[*PATH_KEYS, f"{endpoint}_step"], validate="one_to_one", how="left")
            for short, long in (("Ip", "I_p"), ("Is", "I_s")):
                if candidate[f"approx_{short}"].isna().any():
                    raise ValueError("paired effects require every selected start/end measurement")
                candidate[f"{endpoint}_{long}"] = candidate.pop(f"approx_{short}")
        approximate_pairs = paired_table(candidate)
        keys = ["profile", "task_id", "stage", "contrast"]
        both = baseline_pairs.merge(approximate_pairs, on=keys, suffixes=("_exact", "_approximate"), validate="one_to_one")
        for metric in ("I_p", "I_s", "I_h", "separation", "excess_concordance"):
            rows = both[keys].copy()
            rows["bins"], rows["metric"] = size, metric
            rows["exact_delta"] = both[f"delta_{metric}_exact"]
            rows["approximate_delta"] = both[f"delta_{metric}_approximate"]
            rows["delta_error"] = rows["approximate_delta"] - rows["exact_delta"]
            rows["sign_changed"] = np.sign(rows["approximate_delta"]) != np.sign(rows["exact_delta"])
            paired_rows.append(rows)
        estimates = effect_summaries(approximate_pairs)
        joined = baseline_effects.merge(estimates, on=EFFECT_KEYS, suffixes=("_exact", "_approximate"), validate="one_to_one")
        joined["bins"] = size
        joined["mean_error"] = joined["mean_approximate"] - joined["mean_exact"]
        joined["mean_sign_changed"] = np.sign(joined["mean_approximate"]) != np.sign(joined["mean_exact"])
        # The interval orientation is descriptive; no equivalence threshold is fitted.
        for mode in ("exact", "approximate"):
            finite = np.isfinite(joined[[f"lo_{mode}", f"hi_{mode}"]].to_numpy(float)).all(axis=1)
            side = np.select([joined[f"lo_{mode}"] > 0, joined[f"hi_{mode}"] < 0], [1, -1], default=0).astype(float)
            side[~finite] = np.nan
            joined[f"interval_side_{mode}"] = side
        valid = joined[["interval_side_exact", "interval_side_approximate"]].notna().all(axis=1)
        joined["interval_side_changed"] = pd.Series(pd.NA, index=joined.index, dtype="boolean")
        joined.loc[valid, "interval_side_changed"] = joined.loc[valid, "interval_side_exact"] != joined.loc[valid, "interval_side_approximate"]
        effect_rows.append(joined)
    return pd.concat(paired_rows, ignore_index=True), pd.concat(effect_rows, ignore_index=True), baseline_pairs


def paired_trajectory_effects(paths, baseline_pairs):
    """Compare identical within-task trajectory contrasts, with their actual horizons."""
    rows = []
    if baseline_pairs.empty:
        return pd.DataFrame(columns=[*EFFECT_KEYS, "bins", "quantity", "mode"])
    lookup = paths.set_index([*PATH_KEYS, "bins"])
    for size in sorted(paths["bins"].unique()):
        pairs = baseline_pairs.copy()
        for mode in ("baseline", "fixed", "adaptive"):
            for quantity in ("Iw", "active_step"):
                pairs[f"delta_{mode}_{quantity}"] = [
                    lookup.loc[(p.profile, p.task_id, p.stage, p.treatment, size), f"{mode}_{quantity}"]
                    - lookup.loc[(p.profile, p.task_id, p.stage, p.control, size), f"{mode}_{quantity}"]
                    for p in pairs.itertuples()]
        for scope, requested in intervention_scopes(pairs):
            for key, frame in requested.groupby([*STRATA, "stage", "contrast"], dropna=False, sort=False):
                for quantity in ("Iw", "active_step"):
                    exact = frame[f"delta_baseline_{quantity}"]
                    for mode in ("fixed", "adaptive"):
                        approximate = frame[f"delta_{mode}_{quantity}"]
                        exact_interval,approximate_interval = mean_interval(exact),mean_interval(approximate)
                        error = _error_summary(approximate - exact)
                        rows.append((*key,scope,size,quantity,mode,
                            *(getattr(exact_interval,name) for name in ("n","mean","lo","hi")),
                            *(getattr(approximate_interval,name) for name in ("n","mean","lo","hi")),
                            *(getattr(error,item.name) for item in fields(ErrorSummary)),
                            int((np.sign(exact) != np.sign(approximate)).sum())))
    return pd.DataFrame(rows,columns=[*STRATA,"stage","contrast","scope","bins","quantity","mode",
        *[name+"_exact" for name in ("n","mean","lo","hi")],
        *[name+"_approximate" for name in ("n","mean","lo","hi")],
        *[item.name for item in fields(ErrorSummary)],"pair_sign_changes"])


def benchmark_summary(benchmarks):
    required = ["bins", "repeat", "exact_total_seconds", "prepare_seconds", "evaluate_seconds", "binned_total_seconds"]
    if benchmarks.empty:
        benchmarks = benchmarks.reindex(columns=list(dict.fromkeys([*benchmarks.columns, *required, "same_input_sha"])))
    _required(benchmarks, [*required, "same_input_sha"], "benchmarks")
    result = benchmarks.copy()
    if result["same_input_sha"].isna().any():
        raise ValueError("benchmark omits prepared input identities")
    sizes = set(result["bins"])
    identities = ["case_id"] if "case_id" in result else PATH_KEYS + ["step"] if set([*PATH_KEYS, "step"]) <= set(result) else ["same_input_sha"]
    for _, frame in result.groupby([*identities, "repeat"], sort=False):
        if (set(frame["bins"]) != sizes or frame["bins"].value_counts().nunique() != 1
                or frame["same_input_sha"].nunique() != 1):
            raise ValueError("benchmark KDE grids do not use matched prepared inputs")
    for column in required[2:]:
        if not np.isfinite(result[column].to_numpy(float)).all() or (result[column] < 0).any():
            raise ValueError("benchmark durations must be finite and nonnegative")
    result["end_to_end_speedup"] = result["exact_total_seconds"] / result["binned_total_seconds"].replace(0, np.nan)
    result["prepared_evaluation_speedup"] = np.nan
    if "exact_evaluate_seconds" in result:
        available = result["exact_evaluate_seconds"].dropna()
        if not np.isfinite(available).all() or (available < 0).any():
            raise ValueError("exact density-evaluation durations must be finite and nonnegative")
        result["prepared_evaluation_speedup"] = result["exact_evaluate_seconds"] / result["evaluate_seconds"].replace(0, np.nan)
    rows = []
    for size, frame in result.groupby("bins", sort=False):
        for quantity in [*required[2:], "end_to_end_speedup", "prepared_evaluation_speedup"]:
            values = frame[quantity].dropna()
            rows.append((size,quantity,len(values),values.median(),values.min(),values.max(),values.quantile(.95)))
    return result, pd.DataFrame(rows, columns=["bins", "quantity", "timing_observations", "median", "minimum", "maximum", "p95"])


def figures(point_summary, trajectories, effects, benchmarks, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, stem):
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"{stem}.{suffix}", dpi=190, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for ax, metric in zip(axes, ("I_p", "I_s")):
        frame = point_summary[(point_summary["grid"] == "prepared") & (point_summary["metric"] == metric)].sort_values("bins")
        for column, label in (("median_absolute_error", "median"), ("p95_absolute_error", "95th percentile"), ("maximum_absolute_error", "maximum")):
            ax.plot(frame["bins"], frame[column], "o-", label=label)
        ax.set(xlabel="KDE grid points B", ylabel="Absolute numerical error", title=metric)
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Prepared observations; selected cohort only")
    save(fig, "state_errors")

    quantities = (("fixed_Iw_error", "Fixed-grid $I_w$ error"), ("adaptive_Iw_error", "Adaptive $I_w$ error"),
                  ("grid_Iw_change", "Adaptive minus fixed $I_w$"), ("fixed_active_step_error", "Fixed-grid activity-step error"),
                  ("adaptive_active_step_error", "Adaptive activity-step error"), ("step_count_difference", "Accepted-step count change"))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    sizes = sorted(trajectories["bins"].unique())
    for ax, (quantity, label) in zip(axes.flat, quantities):
        ax.boxplot([trajectories.loc[trajectories["bins"] == size, quantity].to_numpy() for size in sizes], tick_labels=sizes)
        ax.axhline(0, color=".6", lw=.8)
        ax.set(title=label, xlabel="B")
        ax.tick_params(axis="x", labelrotation=35)
    fig.suptitle("Full-prefix trajectory diagnostics; selected cohort only")
    save(fig, "trajectory_errors")

    chosen = effects[(effects["sensitivity"] == "core") & (
        effects["scope"].isin(("early_q_gt_alpha", "middle_P_ge_0.4_E_lt_0.1", "middle_consensus_gains", "middle_remaining"))
        | ((effects["stage"].isin(("late", "full"))) & (effects["scope"] == "all_in_stratum")))]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for ax, metric in zip(axes, ("I_p", "I_s")):
        for key, frame in chosen[chosen["metric"] == metric].groupby(["stage", "contrast", "scope"], sort=False):
            ax.plot(frame["bins"], frame["mean_error"], "o-", label=" / ".join(key))
        ax.axhline(0, color=".6", lw=.8)
        ax.set(title=metric, xlabel="B", ylabel="Error in paired mean effect")
        ax.grid(alpha=.2)
    if not chosen.empty:
        axes[1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))
    save(fig, "paired_effect_errors")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for ax, quantity, label in zip(axes, ("prepared_evaluation_speedup", "end_to_end_speedup"),
                                   ("Same prepared density evaluation", "KDE metric total (incl. preparation)")):
        frame = benchmarks[benchmarks["quantity"] == quantity].sort_values("bins")
        ax.plot(frame["bins"], frame["median"], "o-")
        ax.axhline(1, color=".6", lw=.8)
        ax.set(title=label, xlabel="B", ylabel="Median matched timing ratio")
        ax.grid(alpha=.2)
    save(fig, "kernel_timings")


def markdown_report(points, paths, point_summary, trajectory_summary, effects, benchmarks, provenance):
    tasks = paths[["profile", "task_id"]].drop_duplicates()
    unique_paths = paths[PATH_KEYS].drop_duplicates()
    mode = provenance.configuration.selection.mode
    baseline = provenance.baseline
    text = ["# KDE数值近似对照", "", f"本次队列模式为`{mode}`：{len(tasks)}个任务组、{len(unique_paths)}条路径；冻结完整基线含{baseline.paths}条路径、{baseline.points}个准备观测。每个B使用相同所选任务和全部阶段/政策。本报告描述该队列的数值误差，没有据此宣布科学结论等价或将代表队列外推为全量。", "",
            "只改变距离分布KDE求值。意见/网络状态、已有采样和带宽约定、既有P/E阈值及终态保持来源定义。I_h作为未改变量列出；Ip、Is的正负变化不被解释为干预优劣。", "",
            "## 状态误差", "", "准备观测网格、exact自适应网格、各B自适应网格分开统计；candidate网格未重算exact的点明确计为缺失，未补零。表中观测数包含路径共享状态，不视作独立实验重复。", "",
            "| B | 量 | 准备点数 | 绝对误差中位数 | 95分位 | 最大值 |", "|---:|---|---:|---:|---:|---:|"]
    for r in point_summary[(point_summary["grid"] == "prepared") & point_summary["metric"].isin(("I_p", "I_s"))].itertuples():
        text.append(f"| {r.bins} | {r.metric} | {r.n} | {r.median_absolute_error:.4g} | {r.p95_absolute_error:.4g} | {r.maximum_absolute_error:.4g} |")
    text.extend(["", "## 完整轨迹与配对干预", "", f"exact、fixed和adaptive统一使用endpoint_complement={endpoint_mode(provenance.configuration)}，即{'包含' if endpoint_mode(provenance.configuration) else '不含'}终点补项(1−I_p(T))I_h(T)。fixed保留exact accepted_steps，仅替换该网格上的KDE值；adaptive同时允许既有自适应算法改变accepted_steps。两者都保留0到原终点的完整前缀。adaptive−fixed另列，以区分求值近似和网格改变的贡献。I_s活动时刻沿用calc_active_step及原阈值，从对应曲线重建；16步准备观测没有被冒充为既有I_w采样网格。", "",
                 "| B | 诊断 | 数量 | 误差95分位 | 最大绝对误差 |", "|---:|---|---:|---:|---:|"])
    for r in trajectory_summary[trajectory_summary["quantity"].isin(("fixed_Iw_error", "adaptive_Iw_error", "adaptive_active_step_error"))].itertuples():
        text.append(f"| {r.bins} | {r.quantity} | {r.n} | {r.p95_absolute_error:.4g} | {r.maximum_absolute_error:.4g} |")
    text.extend(["", "端点干预比较直接复用metric_comparison.report中的既有paired_table、effect_summaries及预设子集；P/E和终态原样保留。每个B和exact使用同一所选任务集合；中期转共识与其余任务仍拆开，不能将少数恢复产生的均值写成普遍效应。stage的512步后效窗口与timing共同终点分别保留。", "",
                 "paired_effects.csv保留exact/近似配对均值、原配对t区间、均值方向和区间侧别变化；逐对数值误差另存。接近零的原始差值容易发生符号变化，须结合误差幅度阅读，不能直接称为实质政策逆转。区间变化是数值敏感性描述，没有新增显著性或等价性检验。paired_trajectory_effects.csv另外比较完整轨迹Iw及活动时刻差。", "",
                 "## 计时与来源", "", "prepared求值与KDE指标总耗时（含准备）分别报告。只有存在同一prepared输入的exact_evaluate_seconds时才给prepared求值比；KDE指标总耗时比使用同一状态的exact_total_seconds/binned_total_seconds（CSV列名end_to_end_speedup）。准备耗时另列。此计时只包括准备及Ip/Is求值，不含graph/opinion读取、I_h或完整pipeline，不能外推整轮加速。缓存重验及baseline整轮墙钟耗时不被用作exact KDE耗时，计时重复也不充作科研独立样本。下表速度比是逐个匹配输入计时比的中位数，大于1表示近似较快；不是总时间之比。", "",
                 "| B | prepared求值中位速度比 | KDE指标总耗时（含准备）中位速度比 | 准备耗时中位数(s) | 近似求值中位数(s) |", "|---:|---:|---:|---:|---:|"])
    def timing(size, quantity):
        selected = benchmarks[(benchmarks["bins"] == size) & (benchmarks["quantity"] == quantity)]
        value = selected.iloc[0]["median"] if len(selected) else np.nan
        return f"{value:.5g}" if np.isfinite(value) else "未测"
    for size in sorted(paths["bins"].unique()):
        text.append(f"| {size} | {timing(size, 'prepared_evaluation_speedup')} | {timing(size, 'end_to_end_speedup')} | {timing(size, 'prepare_seconds')} | {timing(size, 'evaluate_seconds')} |")
    if {"extra_exact_calls", "extra_exact_seconds"} <= set(paths):
        shared = paths.drop_duplicates(PATH_KEYS)
        text.extend(["", f"重建exact自适应曲线另补算{int(shared['extra_exact_calls'].sum())}个off-prepared状态，累计{shared['extra_exact_seconds'].sum():.4g}秒；共享路径重建按路径计一次，没有乘以B数量。该成本与公平kernel计时分开。"])
    text.extend(["", "collection/configuration保留实现、输入与重算成本记录；本报告不重跑任何计算。report.json绑定冻结基线、采集配置、本报告源码和全部输出SHA256。", "",
                 "数值偏差的最坏状态见worst_states.csv，完整逐状态误差见point_errors.csv.gz；轨迹新增/删除步数和活动时刻差见trajectory_diagnostics.csv.gz。当前报告不设未经讨论的通过阈值。", ""])
    return "\n".join(text)


def _produce_report(preparation, baseline_report, collection, output, *, make_figures=True, allow_partial=False):
    output = Path(output).resolve()
    points, paths, benchmarks, endpoints, provenance = load_inputs(preparation, baseline_report, collection, allow_partial=allow_partial)
    errors, state_summary, worst = point_errors(points)
    trajectories, trajectory_summary = trajectory_errors(paths)
    pairs, effects, baseline_pairs = paired_effects(points, endpoints)
    trajectory_pairs = paired_trajectory_effects(paths, baseline_pairs)
    timing_rows, timing_summary = benchmark_summary(benchmarks)
    output.mkdir(parents=True,exist_ok=True)
    for name, frame in (("point_errors.csv.gz", errors), ("state_errors.csv", state_summary), ("worst_states.csv", worst),
                        ("trajectory_diagnostics.csv.gz", trajectories), ("trajectory_errors.csv", trajectory_summary),
                        ("paired_point_errors.csv.gz", pairs), ("paired_effects.csv", effects),
                        ("paired_trajectory_effects.csv", trajectory_pairs), ("benchmark_pairs.csv", timing_rows),
                        ("benchmark_summary.csv", timing_summary)):
        write_table(output, name, frame)
        frame.to_csv(output / name, index=False)
    (output / "report.md").write_text(markdown_report(points, paths, state_summary, trajectory_summary, effects, timing_summary, provenance), encoding="utf-8")
    if make_figures:
        figures(state_summary, trajectories, effects, timing_summary, output)
    result = Report(schema=1,status=provenance.collection.status,scope=provenance.configuration.selection,
        endpoint_complement=endpoint_mode(provenance.configuration),
        selected_tasks=len(endpoints[["profile","task_id"]].drop_duplicates()),selected_paths=len(endpoints),
        baseline_paths=provenance.baseline.paths,baseline_points=provenance.baseline.points,
        bins=tuple(sorted(paths["bins"].unique().tolist())),measured_rows=len(points),
        preparation_sha256=sha(Path(preparation)/"preparation.json"),baseline_report_sha256=sha(Path(baseline_report)/"report.json"),
        collection_sha256=sha(Path(collection)/"collection.json"),configuration_sha256=sha(Path(collection)/"configuration.json"),
        report_source_sha256=sha(Path(__file__)),paired_analysis_source_sha256=provenance.baseline.report_source_sha256,
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    atomic_json(output / "report.json",result)
    return result


def report(preparation,baseline_report,collection,output,*,make_figures=True,allow_partial=False):
    output = Path(output).resolve()
    load_inputs(preparation,baseline_report,collection,allow_partial=allow_partial)
    inputs = ReportInputs(sha(Path(preparation)/"preparation.json"),sha(Path(collection)/"collection.json"),
        {str(Path(inspect.getfile(mean_interval)).resolve()):sha(inspect.getfile(mean_interval)),
         str(Path(__file__).resolve()):sha(__file__),str(Path(baseline_report)/"report.json"):sha(Path(baseline_report)/"report.json")},
        make_figures,allow_partial)
    if output.exists() and any(output.iterdir()) and not (output/"report_inputs.json").is_file():
        raise ValueError("report output already exists without matching inputs")
    output.mkdir(parents=True,exist_ok=True)
    with session_lock(output/"runtime"):
        freeze_manifest(output/"report_inputs.json",inputs)
        def produce(stage):
            _produce_report(preparation,baseline_report,collection,stage,make_figures=make_figures,allow_partial=allow_partial)
        artifact = ArtifactCache(output/"runtime/cache").materialize("report",inputs,produce)
        for path in artifact.iterdir():
            if path.is_file() and path.name != "artifact.json":
                shutil.copy2(path,output/path.name)
        return load_contract(output/"report.json",Report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", required=True, type=Path)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    result = report(args.preparation, args.baseline_report, args.collection, args.output,
                    make_figures=not args.no_figures, allow_partial=args.allow_partial)
    print(json.dumps({key: value for key, value in to_wire(result).items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
