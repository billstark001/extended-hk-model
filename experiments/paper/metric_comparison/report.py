"""Report matched states, existing pathway labels and paired intervention effects."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.experiments.plans import read_plan
from common.io.json import atomic_json, fingerprint, read_result, sha
from ehk.metrics.pathway import (
    EXCESS_THRESHOLD,
    OBSERVATION_INTERVAL,
    PATHWAY_LABELS,
    SEPARATION_THRESHOLD,
    first_passage,
)
from ehk.metrics.polarization import opinion_peak_count
from experiments.paper.recommendation_shared.analysis import mean_interval, outcome

from .core import (PATH_KEYS, POINT_KEYS, _arms, _authenticate_files, _bundle_layout, authenticate_preparation,
                   endpoint_mode, validate_endpoint_column)

STATE_METRICS = ("separation", "I_p", "excess_concordance", "I_h", "I_s", "edge_concordance", "pair_concordance")
LABELS = {"separation": "P", "I_p": "$I_p$", "excess_concordance": "E", "I_h": "$I_h$",
          "I_s": "$I_s$", "edge_concordance": "$C_{edge}$", "pair_concordance": "$C_{pair}$"}
STRATA = ["profile", "sensitivity", "epsilon", "history"]
CONTRASTS = (("staged", "switch", "hold"), ("staged", "bridge", "switch"),
             ("staged", "hide_reposts", "switch"), ("staged_timing", "state", "fixed"))
START_VALIDATION_EPS = 64 * np.finfo(np.float64).eps


def _csv(path):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def load_aligned(preparation, comparison, *, allow_partial=False):
    """Authenticate the input generation; no metric calculation occurs here."""
    preparation, comparison = Path(preparation).resolve(), Path(comparison).resolve()
    manifest = authenticate_preparation(preparation)
    compared = json.loads((comparison / "comparison.json").read_text())
    _authenticate_files(comparison, compared)
    if compared["preparation_sha256"] != sha(preparation / "preparation.json"):
        raise ValueError("comparison belongs to another preparation")
    if compared["status"] != "complete" and not allow_partial:
        raise ValueError("legacy metrics remain pending; use --allow-partial only for a labeled interim report")
    points, paths = (_csv(comparison / name) for name in ("points_aligned.csv.gz", "paths_aligned.csv.gz"))
    policy = endpoint_mode(compared)
    # Pending rows have no I_w and therefore no claimed endpoint policy.
    available = paths["I_w"].notna() if "I_w" in paths else pd.Series(False, index=paths.index)
    validate_endpoint_column(paths.loc[available].copy(), policy)
    paths["endpoint_complement"] = policy
    if points.duplicated(POINT_KEYS).any() or paths.duplicated(PATH_KEYS).any():
        raise ValueError("duplicate aligned state or trajectory identity")
    if len(points) != manifest["points"] or len(paths) != manifest["paths"]:
        raise ValueError("aligned tables differ from the prepared cohort")
    if not set(STATE_METRICS) <= set(points) or not {"I_w", "terminal_json", "start_step", "end_step"} <= set(paths):
        raise ValueError("aligned tables omit required state or trajectory fields")
    if not allow_partial and (not np.isfinite(points[list(STATE_METRICS)].to_numpy(float)).all()
                              or not np.isfinite(paths["I_w"].to_numpy(float)).all()):
        raise ValueError("complete comparison contains missing/nonfinite legacy metrics")
    return points, paths, {"preparation": manifest, "comparison": compared}


def endpoint_table(points, paths):
    """Resolve exact start/end observations without interpolation or horizon pooling."""
    result = paths.copy()
    for endpoint in ("start", "end"):
        selected = points[[*PATH_KEYS, "step", *STATE_METRICS]].rename(
            columns={"step": f"{endpoint}_step", **{name: f"{endpoint}_{name}" for name in STATE_METRICS}}
        )
        result = result.merge(selected, on=[*PATH_KEYS, f"{endpoint}_step"], how="left", validate="one_to_one")
    if result[[f"{end}_{name}" for end in ("start", "end")
               for name in ("separation", "excess_concordance")]].isna().any().any():
        raise ValueError("a prepared trajectory endpoint has no exact recorded observation")
    result["followup_steps"] = result["end_step"] - result["start_step"]
    result["terminal_outcome"] = result["terminal_json"].map(lambda text: outcome({"terminal": json.loads(text)}))
    return result


def endpoint_features(preparation, endpoints):
    """Count existing KDE modes on authenticated saved final opinions, including references."""
    manifest = authenticate_preparation(preparation)
    if not manifest.get("sources"):
        raise ValueError("endpoint peak counts require authenticated local summary sources")
    if endpoints.duplicated(PATH_KEYS).any():
        raise ValueError("duplicate endpoint identity for KDE feature join")
    requested = {tuple(row[name] for name in PATH_KEYS): row for row in endpoints.to_dict("records")}
    rows = []
    source_sha = sha(Path(inspect.getfile(opinion_peak_count)))
    for source in manifest["sources"]:
        bundle = Path(source["bundle"])
        data, study, manifest_path, source_manifest = _bundle_layout(bundle)
        if sha(manifest_path) != source["manifest_sha256"]:
            raise ValueError("endpoint summary manifest differs from the prepared generation")
        plan = read_plan(study / "plan.json")
        profile = plan["profile"]
        if plan["plan_fingerprint"] != source["plan_fingerprint"] or profile != source["profile"]:
            raise ValueError("endpoint summary plan differs from the prepared cohort")
        archived = {}
        archived_table = ""
        relative_table = "tables/runs.csv.gz"
        if relative_table in source_manifest["files"]:
            archived_table = str(data / relative_table)
            saved = _csv(archived_table)
            required = ["task_id", "stage", "arm", "end_step", "opinion_peaks"]
            if not set(required) <= set(saved) or saved.duplicated(required[:3]).any():
                raise ValueError("authenticated endpoint count table has ambiguous identities")
            archived = {tuple([profile, row["task_id"], row["stage"], row["arm"]]): row
                        for row in saved.to_dict("records")}
        for task in plan["tasks"]:
            record_path = study / "records" / (task["id"] + ".json.gz")
            if not record_path.is_file():
                continue
            record = read_result(record_path)
            if record["task_fingerprint"] != fingerprint(task):
                raise ValueError("endpoint summary does not match its frozen task")
            record_sha = sha(record_path)
            for stage, arm, summary, _ in _arms(profile, record["result"]):
                key = (profile, task["id"], stage, arm)
                if key not in requested:
                    raise ValueError("authenticated endpoint summary is outside the prepared cohort")
                path = requested[key]
                end_step = summary["endpoints"][-1]["step"]
                if "plan_fingerprint" in path and path["plan_fingerprint"] != plan["plan_fingerprint"]:
                    raise ValueError("endpoint count plan identity differs from its prepared trajectory")
                if end_step != path["end_step"] or summary.get("stopped_at", end_step) != end_step:
                    raise ValueError("endpoint count horizon differs from its prepared trajectory")
                if "raw_record" in path and summary["raw_record"] != path["raw_record"]:
                    raise ValueError("endpoint count raw identity differs from its prepared trajectory")
                opinions = np.asarray(summary["final_opinions"], dtype=np.float64)
                if opinions.ndim != 1 or len(opinions) != task["metadata"]["NodeCount"] or not np.isfinite(opinions).all():
                    raise ValueError("endpoint summary final opinions are missing, nonfinite or wrong-sized")
                count = int(opinion_peak_count(opinions))
                previous = archived.get(key)
                saved_count = np.nan
                if previous is not None:
                    if previous["end_step"] != end_step:
                        raise ValueError("authenticated saved peak count has a different endpoint horizon")
                    if "raw_record" in previous and previous["raw_record"] != summary["raw_record"]:
                        raise ValueError("authenticated saved peak count has a different raw identity")
                    saved_count = previous["opinion_peaks"]
                    if not np.isfinite(saved_count) or saved_count < 1 or saved_count % 1:
                        raise ValueError("authenticated saved peak count is not a positive integer")
                rows.append({**dict(zip(PATH_KEYS, key)), "end_step": end_step,
                             "opinion_peaks": count, "saved_opinion_peaks": saved_count,
                             "saved_count_matches": count == saved_count if previous is not None else None,
                             "summary_sha256": record_sha, "summary_path": str(record_path),
                             "peak_implementation_sha256": source_sha,
                             "saved_count_table": archived_table if previous is not None else ""})
    features = pd.DataFrame(rows)
    if len(features) != len(endpoints) or features.duplicated(PATH_KEYS).any():
        raise ValueError("authenticated final-opinion summaries do not cover the prepared endpoint cohort exactly")
    return features


def attach_endpoint_features(endpoints, features):
    if len(features) != len(endpoints) or features.duplicated(PATH_KEYS).any():
        raise ValueError("endpoint features must cover exactly one row per prepared path")
    result = endpoints.merge(features, on=[*PATH_KEYS, "end_step"], how="left", validate="one_to_one")
    if result["opinion_peaks"].isna().any():
        raise ValueError("an endpoint lacks a peak count at the exact prepared horizon")
    return result


def peak_summaries(endpoints):
    """Cross-tabulate two existing endpoint descriptions without assigning a truth label."""
    keys = [*STRATA, "stage", "arm", "start_step", "end_step", "opinion_peaks"]
    cross = endpoints.groupby([*keys, "terminal_outcome"], dropna=False, sort=False).size().reset_index(name="paths")
    rows = []
    for key, frame in endpoints.groupby(keys, dropna=False, sort=False):
        matched = frame[frame[["end_I_p", "end_I_h"]].notna().all(axis=1)]
        for metric in ("I_p", "I_h", "separation", "excess_concordance"):
            values = matched[f"end_{metric}"]
            rows.append({**dict(zip(keys, key)), "metric": metric, "requested_paths": len(frame),
                         "matched_paths": len(matched), "mean": values.mean(), "median": values.median(),
                         "minimum": values.min(), "maximum": values.max()})
    return cross, pd.DataFrame(rows)


def checkpoint_table(endpoints):
    """Use each task/stage's hold arm once, without duplicating shared starts."""
    selected = endpoints[(endpoints["arm"] == "hold") & (endpoints["profile"].isin(["staged", "staged_timing"]))]
    identifiers = [*PATH_KEYS, *STRATA[1:], "influence", "rewiring", "repost", "start_step"]
    provenance = [name for name in ("plan_fingerprint", "binary_sha256", "raw_record") if name in selected]
    result = selected[[*identifiers, *provenance, *[f"start_{name}" for name in STATE_METRICS]]].rename(
        columns={"start_step": "step", **{f"start_{name}": name for name in STATE_METRICS}}
    )
    if result.duplicated(["profile", "task_id", "stage"]).any():
        raise ValueError("duplicate canonical hold checkpoint")
    return result.reset_index(drop=True)


def checkpoint_summaries(checkpoints):
    keys = [*STRATA, "stage", "step"]
    rows = []
    for key, frame in checkpoints.groupby(keys, dropna=False, sort=False):
        for metric in STATE_METRICS:
            values = frame[metric].dropna()
            rows.append({**dict(zip(keys, key)), "metric": metric, "checkpoints": len(frame),
                         "available": len(values), "mean": values.mean(), "median": values.median(),
                         "minimum": values.min(), "maximum": values.max()})
    return pd.DataFrame(rows)


def pathway_table(points, endpoints):
    """Apply the existing first-passage implementation, retaining staged prefixes."""
    groups = {key: frame.sort_values("step") for key, frame in points.groupby(PATH_KEYS, sort=False)}
    rows = []
    for path in endpoints.to_dict("records"):
        key = tuple(path[name] for name in PATH_KEYS)
        window = groups[key]
        full = window
        if path["profile"] == "staged" and path["stage"] != "reference":
            reference_key = ("staged", path["task_id"], "reference", "reference")
            if reference_key not in groups:
                raise ValueError("staged first-passage classification requires the shared reference prefix")
            reference = groups[reference_key]
            checkpoint = reference[reference["step"] == path["start_step"]]
            if len(checkpoint) != 1:
                raise ValueError("reference lacks the exact intervention checkpoint")
            for name in ("separation", "excess_concordance"):
                if checkpoint.iloc[0][name] != window.iloc[0][name]:
                    raise ValueError("staged branch and reference disagree at the intervention checkpoint")
            full = pd.concat([reference[reference["step"] < path["start_step"]], window], ignore_index=True)
        steps = full["step"].to_numpy(int)
        if steps[0] != 0 or steps[-1] != path["end_step"] or np.any(np.diff(steps) != OBSERVATION_INTERVAL):
            raise ValueError("first-passage comparison requires the complete original 16-step grid")
        passage = first_passage(full[["step", "separation", "excess_concordance"]].itertuples(index=False, name=None))
        rows.append({**{name: path[name] for name in (*PATH_KEYS, *STRATA[1:], "influence", "rewiring", "repost",
                                                   "start_step", "end_step", "followup_steps", "I_w", "terminal_outcome")},
                     "pathway": passage.label, "separation_step": passage.separation_step,
                     "segregation_step": passage.segregation_step,
                     "prefix_observations": len(full) - len(window)})
    return pd.DataFrame(rows)


def state_summaries(endpoints):
    """Describe endpoint states by design stratum; repeated time points are not replicates."""
    keys = [*STRATA, "stage", "arm", "start_step", "end_step", "terminal_outcome"]
    rows = []
    for key, frame in endpoints.groupby(keys, dropna=False, sort=False):
        for metric in STATE_METRICS:
            values = frame[f"end_{metric}"].dropna()
            rows.append({**dict(zip(keys, key)), "metric": metric, "paths": len(frame), "available": len(values),
                         "mean": values.mean(), "median": values.median(), "minimum": values.min(), "maximum": values.max()})
    return pd.DataFrame(rows)


def pathway_summaries(paths):
    keys = [*STRATA, "stage", "arm", "start_step", "end_step", "pathway"]
    rows = []
    for key, frame in paths.groupby(keys, dropna=False, sort=False):
        values = frame["I_w"].dropna()
        rows.append({**dict(zip(keys, key)), "paths": len(frame), "available_I_w": len(values),
                     "I_w_mean": values.mean(), "I_w_median": values.median(),
                     "I_w_minimum": values.min(), "I_w_maximum": values.max()})
    return pd.DataFrame(rows)


def paired_table(endpoints):
    """Keep contrasts within task, stage and identical absolute follow-up horizons."""
    rows = []
    for profile, treatment, control in CONTRASTS:
        selected = endpoints[endpoints["profile"] == profile]
        keys = ["task_id", "stage"]
        left, right = (selected[selected["arm"] == arm].set_index(keys) for arm in (treatment, control))
        paired = left.join(right, how="inner", lsuffix="_treatment", rsuffix="_control")
        for key, pair in paired.iterrows():
            for name in ("start_step", "end_step", "sensitivity", "epsilon", "history", "influence", "rewiring", "repost"):
                if pair[name + "_treatment"] != pair[name + "_control"]:
                    raise ValueError(f"paired interventions differ in {name}")
            for metric in ("separation", "excess_concordance", "I_p", "I_h", "I_s"):
                a, b = pair[f"start_{metric}_treatment"], pair[f"start_{metric}_control"]
                tolerance = START_VALIDATION_EPS * max(1.0, abs(a), abs(b)) if metric.startswith("I_") else 0.0
                if np.isfinite(a) and np.isfinite(b) and abs(a - b) > tolerance:
                    raise ValueError(f"paired interventions start at different {metric} states")
            row = {"profile": profile, "task_id": key[0], "stage": key[1], "treatment": treatment,
                   "control": control, "contrast": f"{treatment}_minus_{control}",
                   **{name: pair[name + "_control"] for name in (*STRATA[1:], "influence", "rewiring", "repost",
                                                                "start_step", "end_step", "followup_steps")}}
            for metric in STATE_METRICS:
                row[f"start_{metric}"] = pair[f"start_{metric}_control"]
                row[f"start_difference_{metric}"] = pair[f"start_{metric}_treatment"] - pair[f"start_{metric}_control"]
                row[f"control_{metric}"] = pair[f"end_{metric}_control"]
                row[f"treatment_{metric}"] = pair[f"end_{metric}_treatment"]
                row[f"delta_{metric}"] = row[f"treatment_{metric}"] - row[f"control_{metric}"]
            row["control_outcome"] = pair["terminal_outcome_control"]
            row["treatment_outcome"] = pair["terminal_outcome_treatment"]
            row["consensus_gain"] = row["treatment_outcome"] == "consensus" and row["control_outcome"] != "consensus"
            row["consensus_loss"] = row["treatment_outcome"] != "consensus" and row["control_outcome"] == "consensus"
            row["legacy_pair_available"] = all(np.isfinite(row[f"delta_{name}"]) for name in STATE_METRICS)
            rows.append(row)
    return pd.DataFrame(rows)


def intervention_scopes(pairs):
    """Prespecified existing design/state subsets; no thresholds are fitted."""
    if pairs.empty:
        return
    for key, frame in pairs.groupby([*STRATA, "stage", "contrast"], dropna=False, sort=False):
        yield "all_in_stratum", frame
    core = pairs[pairs["sensitivity"] == "core"]
    ordinary = core[(core["profile"] == "staged") & (core["contrast"] == "switch_minus_hold")]
    early = ordinary[(ordinary["stage"] == "early") & (ordinary["rewiring"] > ordinary["influence"])]
    middle = ordinary[(ordinary["stage"] == "middle") & (ordinary["start_separation"] >= SEPARATION_THRESHOLD)
                      & (ordinary["start_excess_concordance"] < EXCESS_THRESHOLD)]
    if not early.empty:
        yield "early_q_gt_alpha", early
    if not middle.empty:
        yield "middle_P_ge_0.4_E_lt_0.1", middle
        yield "middle_consensus_gains", middle[middle["consensus_gain"]]
        yield "middle_remaining", middle[~middle["consensus_gain"]]


def effect_summaries(pairs, *, by_cell=False):
    rows = []
    for scope, requested in intervention_scopes(pairs):
        keys = [*STRATA, "stage", "contrast"] + (["influence", "rewiring", "repost"] if by_cell else [])
        for key, group in requested.groupby(keys, dropna=False, sort=False):
            matched = group[group["legacy_pair_available"]]
            for metric in STATE_METRICS:
                values = matched[f"delta_{metric}"]
                rows.append({**dict(zip(keys, key)), "scope": scope, "metric": metric,
                             "requested_pairs": len(group), "pending_pairs": len(group) - len(matched),
                             **mean_interval(values), "sum_delta": values.sum() if len(values) else np.nan,
                             "negative": int((values < 0).sum()), "zero": int((values == 0).sum()),
                             "positive": int((values > 0).sum()),
                             "consensus_gains": int(matched["consensus_gain"].sum()),
                             "consensus_losses": int(matched["consensus_loss"].sum()),
                             "start_steps": json.dumps(sorted(group["start_step"].unique().tolist())),
                             "end_steps": json.dumps(sorted(group["end_step"].unique().tolist())),
                             "followup_steps": json.dumps(sorted(group["followup_steps"].unique().tolist()))})
    return pd.DataFrame(rows)


def _save(fig, output, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=190, bbox_inches="tight")


def figures(endpoints, pathways, effects, output, checkpoints=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none"})
    endpoints = endpoints[endpoints["sensitivity"] == "core"]
    pathways = pathways[pathways["sensitivity"] == "core"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (x, y) in zip(axes.flat, (("I_p", "separation"), ("I_h", "excess_concordance"),
                                   ("I_h", "edge_concordance"), ("I_h", "pair_concordance"))):
        for count, frame in endpoints.groupby("opinion_peaks", sort=True):
            frame = frame.dropna(subset=[f"end_{x}", f"end_{y}"])
            if frame.empty:
                continue
            color = plt.get_cmap("tab10")((int(count) - 1) % 10)
            ax.scatter(frame[f"end_{x}"], frame[f"end_{y}"], s=7, alpha=.3, color=color,
                       label=f"{int(count)} peak(s)", rasterized=True)
        ax.set(xlabel=LABELS[x], ylabel=LABELS[y])
        ax.grid(alpha=.15)
    axes.flat[0].legend(title="Existing opinion KDE peak count", fontsize=7)
    fig.suptitle("Core matched endpoint states; each path retains its own endpoint time")
    _save(fig, output, "matched_states")
    plt.close(fig)
    if checkpoints is not None:
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
        for row, stage in enumerate(("middle", "late")):
            requested = checkpoints[(checkpoints["profile"] == "staged") & (checkpoints["sensitivity"] == "core")
                                    & (checkpoints["stage"] == stage)]
            available = requested[requested[["I_p", "I_h"]].notna().all(axis=1)]
            opportunity = ((available["separation"] >= SEPARATION_THRESHOLD)
                           & (available["excess_concordance"] < EXCESS_THRESHOLD)) if stage == "middle" else pd.Series(False, index=available.index)
            for ax, x, y in ((axes[row, 0], "I_p", "separation"), (axes[row, 1], "I_h", "excess_concordance")):
                for mask, color, label in ((~opportunity, ".55", "Other checkpoints"),
                                           (opportunity, "#0072b2", "P ≥ .4 and E < .1")):
                    subset = available[mask]
                    if not subset.empty:
                        ax.scatter(subset[x], subset[y], s=13, alpha=.55, color=color, label=label)
                ax.axhline(SEPARATION_THRESHOLD if y == "separation" else EXCESS_THRESHOLD, color=".65", ls="--", lw=.8)
                ax.set(xlabel=LABELS[x], ylabel=LABELS[y], title=f"{stage}: {len(available)}/{len(requested)} checkpoints")
                ax.grid(alpha=.15)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Core intervention starts; one canonical hold state per task and stage")
        _save(fig, output, "checkpoint_states")
        plt.close(fig)
    groups = pathways[((pathways["profile"] == "staged") & (pathways["stage"] == "reference"))
                      | (pathways["profile"] == "staged_timing")]
    if not groups.empty:
        scopes = list(groups.groupby(["profile", "arm"], sort=False))
        fig, axes = plt.subplots(1, len(scopes), figsize=(max(5, len(scopes) * 2.5), 4), squeeze=False, constrained_layout=True)
        for ax, (key, frame) in zip(axes.flat, scopes):
            labels = [label for label in PATHWAY_LABELS if (frame["pathway"] == label).any()]
            samples = [frame.loc[frame["pathway"] == label, "I_w"].dropna().to_numpy() for label in labels]
            nonempty = [(label, values) for label, values in zip(labels, samples) if len(values)]
            if nonempty:
                ax.boxplot([v for _, v in nonempty], tick_labels=[f"{k}\nn={len(v)}" for k, v in nonempty], showfliers=True)
            ax.set(title=f"{key[0]}\n{key[1]}", ylabel="$I_w$")
            ax.tick_params(axis="x", labelrotation=55)
        fig.suptitle("Core continuous $I_w$ and existing P/E first-passage labels (fixed 16-step grid)")
        _save(fig, output, "pathway_metrics")
        plt.close(fig)
    chosen = effects[(effects["sensitivity"] == "core") & (effects["scope"].isin(
        ["early_q_gt_alpha", "middle_P_ge_0.4_E_lt_0.1"]) | ((effects["stage"].isin(["late", "full"]))
         & (effects["scope"] == "all_in_stratum")))] if not effects.empty else effects
    if not chosen.empty:
        identifiers = chosen[["scope", "profile", "stage", "contrast"]].drop_duplicates().to_records(index=False).tolist()
        fig, axes = plt.subplots(1, 4, figsize=(12, max(3.5, len(identifiers) * .45)), sharey=True, constrained_layout=True)
        for ax, metric in zip(axes, ("separation", "I_p", "excess_concordance", "I_h")):
            for i, identity in enumerate(identifiers):
                selected = chosen[(chosen["metric"] == metric)]
                for field, value in zip(("scope", "profile", "stage", "contrast"), identity):
                    selected = selected[selected[field] == value]
                if len(selected) == 1 and selected.iloc[0]["n"]:
                    row = selected.iloc[0]
                    ax.plot(row["mean"], i, "o", color="#0072b2")
                    if pd.notna(row["lo"]) and pd.notna(row["hi"]):
                        ax.plot([row["lo"], row["hi"]], [i, i], color="#0072b2")
            ax.axvline(0, color=".6", lw=.8)
            ax.set(xlabel=f"Paired endpoint Δ {LABELS[metric]}", yticks=range(len(identifiers)))
        axes[0].set_yticklabels([f"{a}\n{c}: {d}" for a, b, c, d in identifiers], fontsize=7)
        fig.suptitle("Prespecified contrasts; 95% paired t intervals within each subset")
        _save(fig, output, "intervention_effects")
        plt.close(fig)


def markdown_report(endpoints, pathways, effects, provenance, peak_audit=None):
    available = int(endpoints["end_I_p"].notna().sum())
    text = ["# 同一干预轨迹上的两套指标比较", "",
            f"报告读取 {len(endpoints)} 条已准备路径；其中 {available} 条具有终点旧指标。来源比较状态：`{provenance['comparison']['status']}`。",
            "", "## 比较对象与边界", "",
            "状态图只比较同一节点/网络状态下的指标，不把它们解释为可互换量。I_h 对有正出度用户等权；C_edge 对有向边等权。E=C_edge−C_pair 使用当前意见构成作基线。共识与多个内部同信群体都可有较高 I_h，这不构成 I_h 失效。",
            "", "旧指标指原有指标族，本报告采用本次采集记录的现有实现；它不自动复原早期档案中使用过的不同基线、计数对象或末点补项。主图只使用核心条件；敏感性条件在CSV中单列，未与核心合并。",
            "", "现有I_p是相对随机意见距离分布的JS比，不保证在共识时归零，也不保证小于等于1；因此ΔI_p不能直接替代ΔP或已记录的共识恢复，正负号不自动表示干预优劣。图和表保留原计算值，不裁剪到[0,1]。",
            "", "端点主图按正文已有的KDE意见峰数着色：I_h保留同信性职责，峰数补充意见分布形态；两者的组合不需由I_h独自承担共识/多群区分。KDE峰数与置信分量终态在交叉表中并列，前者使用现有opinion_peak_count，受既有带宽与峰检测规则影响；后者沿用原记录，取决于置信间距、直径与群体质量条件。两者互不充当真值。逐峰数的Ip/Ih/P/E概览使用同一批两套状态量均齐备的端点。",
            "", "检查点状态另用每个task/stage的hold分支起点表示一次，避免把多个分支的共同起点当作多个样本。中/晚期起点图不按未来终态着色；其中P=.4和E=.1仅沿用既有状态阈值。晚期仍由原活动判据确定，阈值线仅供状态参照。配对起点的旧指标允许64倍float64机器epsilon的舍入差（按数值尺度缩放），P/E仍要求精确一致；这只用于验证，不改动保存值。逐对起点差保留在配对CSV。",
            "", f"I_w 保留已有自适应采样与完整0至终点的积分；endpoint_complement={endpoint_mode(provenance['comparison'])}，表示{'包含' if endpoint_mode(provenance['comparison']) else '不含'}终点补项(1−I_p(T))I_h(T)。路径标签使用已有16步观测、P=.4与E=.1的首次越阈顺序，未调阈值、未按I_w=.6另造分类。simultaneous表示同一次观测越阈。staged分支保留共享前缀，标签与该分支的实际终点对应。",
            "", "每个stage的切换结果在该检查点后512步评价，绝对终点可不同；timing的state−fixed使用相同完整终点。不同stage的I_w或终点效应不混作同一时机实验。敏感性条件与核心设计分开；用于比较旧/新终点效应的任务集合完全相同。", "",
            "## 预设干预子集", "", "| 子集 | 比较 | 配对数 | ΔP [95%区间] | ΔI_p [95%区间] | ΔE [95%区间] | ΔI_h [95%区间] |", "|---|---|---:|---|---|---|---|---|"]
    def estimate(row):
        if row.empty or not row.iloc[0]["n"]:
            return "pending"
        r = row.iloc[0]
        return f"{r['mean']:+.5f} [{r['lo']:+.5f}, {r['hi']:+.5f}]" if pd.notna(r["lo"]) else f"{r['mean']:+.5f}"
    if not effects.empty:
        selected = effects[(effects["sensitivity"] == "core") & ((effects["scope"].isin(
            ["early_q_gt_alpha", "middle_P_ge_0.4_E_lt_0.1", "middle_consensus_gains", "middle_remaining"]))
            | ((effects["stage"].isin(["late", "full"])) & (effects["scope"] == "all_in_stratum")))]
        for (scope, stage, contrast), rows in selected.groupby(["scope", "stage", "contrast"], sort=False):
            p, ip = rows[rows["metric"] == "separation"], rows[rows["metric"] == "I_p"]
            e, ih = rows[rows["metric"] == "excess_concordance"], rows[rows["metric"] == "I_h"]
            n = int(p.iloc[0]["n"]) if not p.empty else 0
            text.append(f"| {scope} ({stage}) | {contrast} | {n} | {estimate(p)} | {estimate(ip)} | {estimate(e)} | {estimate(ih)} |")
    text.extend(["", "中期P≥.4、E<.1子集另外拆成已有置信分量分类下的转共识任务和其余任务。`intervention_effects.csv` 保留各组的差值总和、配对数和均值，可判断平均效应是否由少数大幅恢复主导；不把少数案例写成普遍改善。",
                 "", "旧/新指标的变化方向不一致时，应回看同一任务的意见构成与网络状态。较低P、较低I_p和较低E分别回答不同问题，报告没有通过新标量或标签一致率将它们合并。当前数据只支持所列采样社区、参数和观察窗口中的比较。",
                 "", "## 输出与来源", "", "- `endpoint_states.csv.gz`：逐路径端点、参数和终态。",
                 "- `endpoint_features.csv.gz`：来自已认证本地summary.final_opinions的既有KDE峰数、原表复核值与来源。参考轨迹也逐条计算。",
                 "- `kde_confidence_crosstab.csv`、`kde_stratified_states.csv`：已有峰数与置信终态的交叉计数、按峰数分层的Ip/Ih/P/E概览；没有准确率或真值标签。",
                 "- `checkpoint_states.csv`、`checkpoint_summary.csv`：每task/stage一次的hold起点及分层概览。",
                 "- `state_summary.csv`：按研究/参数条件/阶段/政策/终态及实际时间分层的状态概览。",
                 "- `pathway_comparison.csv.gz`、`pathway_summary.csv`：既有首达标签与连续I_w，含实际起止时刻。",
                 "- `paired_interventions.csv.gz`、`intervention_effects.csv`、`intervention_cells.csv`：严格成对的干预对照及预设子集。",
                 "- 科学图提供PNG/PDF/SVG，`report.json`记录输入及输出哈希。", ""])
    if peak_audit is not None:
        text.extend([f"峰数来源复核：计算 {peak_audit['computed_paths']} 条端点；其中 {peak_audit['archived_comparisons']} 条与认证旧runs表逐条比较，{peak_audit['archived_differences']} 条不同。当前helper值和旧表值分别保留，不把差异改写为分类准确率。", ""])
    return "\n".join(text)


def report(preparation, comparison, output, *, make_figures=True, allow_partial=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("report output already exists; choose a new directory")
    points, paths, provenance = load_aligned(preparation, comparison, allow_partial=allow_partial)
    endpoints = endpoint_table(points, paths)
    features = endpoint_features(preparation, endpoints)
    endpoints = attach_endpoint_features(endpoints, features)
    peak_cross, peak_states = peak_summaries(endpoints)
    audited = features[features["saved_count_matches"].notna()]
    peak_audit = {"computed_paths": len(features), "archived_comparisons": len(audited),
                  "archived_differences": int((~audited["saved_count_matches"].astype(bool)).sum())}
    checkpoints = checkpoint_table(endpoints)
    pathways = pathway_table(points, endpoints)
    paired = paired_table(endpoints)
    effects, cells = effect_summaries(paired), effect_summaries(paired, by_cell=True)
    output.mkdir(parents=True)
    for name, frame in (("endpoint_states.csv.gz", endpoints), ("state_summary.csv", state_summaries(endpoints)),
                        ("endpoint_features.csv.gz", features), ("kde_confidence_crosstab.csv", peak_cross),
                        ("kde_stratified_states.csv", peak_states),
                        ("checkpoint_states.csv", checkpoints), ("checkpoint_summary.csv", checkpoint_summaries(checkpoints)),
                        ("pathway_comparison.csv.gz", pathways), ("pathway_summary.csv", pathway_summaries(pathways)),
                        ("paired_interventions.csv.gz", paired), ("intervention_effects.csv", effects),
                        ("intervention_cells.csv", cells)):
        frame.to_csv(output / name, index=False)
    (output / "report.md").write_text(markdown_report(endpoints, pathways, effects, provenance, peak_audit))
    if make_figures:
        figures(endpoints, pathways, effects, output, checkpoints)
    result = {"schema": 1, "status": provenance["comparison"]["status"], "paths": len(paths),
              "endpoint_complement": endpoint_mode(provenance["comparison"]),
              "points": len(points), "paired_interventions": len(paired),
              "endpoint_peak_count_audit": peak_audit,
              "preparation_sha256": sha(Path(preparation) / "preparation.json"),
              "comparison_sha256": sha(Path(comparison) / "comparison.json"),
              "report_source_sha256": sha(Path(__file__)),
              "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    atomic_json(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    result = report(args.preparation, args.comparison, args.output,
                    make_figures=not args.no_figures, allow_partial=args.allow_partial)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
