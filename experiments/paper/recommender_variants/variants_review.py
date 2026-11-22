"""Prepare and compose portable OpinionRandom factorial review figures."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.experiments.plans import read_plan
from common.io.json import atomic_json, sha
from common.plotting import setup_paper_params
from ehk.metrics.pathway import PATHWAY_LABELS
from experiments.paper.recommendation_shared import analysis as shared_analysis
from experiments.paper.recommendation_shared import bundle as shared_bundle
from experiments.paper.recommendation_shared import plot as shared_plot

from .analysis import FACTORIAL_CONTRASTS, opinion_random_factorial

SCHEMA = 1
FACTORS = ((1, 0), (4, 0), (1, 2), (4, 2))
CONTRAST_LABELS = ("ζ: 4 − 1, σ=0", "ζ: 4 − 1, σ=2", "σ: 2 − 0, ζ=1",
                   "σ: 2 − 0, ζ=4", "Interaction")
PATHWAY_COLORS = ("#377eb8", "#e41a1c", "#984ea3", "#4daf4a", "#ff7f00", "#aaaaaa", "#555555")


def _rows(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def matched_factorial_runs(runs):
    """Use the same initial states for all four configurations and M9 reference."""
    if runs.duplicated(["task_id", "arm"]).any():
        raise ValueError("duplicate task/arm observations")
    random = runs[runs["recommender"] == "OpinionRandom"]
    if random.duplicated(["task_id", "zeta", "score_noise"]).any():
        raise ValueError("duplicate factorial configuration for one initial state")
    ids = {
        identifier for identifier, rows in random.groupby("task_id")
        if set(zip(rows["zeta"], rows["score_noise"])) == set(FACTORS)
    }
    selected = runs[runs["task_id"].isin(ids)].copy()
    for _, rows in selected.groupby("task_id"):
        if any(rows[name].nunique(dropna=False) != 1 for name in ("start_separation", "start_excess")):
            raise ValueError("factorial arms have different initial pathway metrics")
    return selected


def _write_matched_tables(output, matched, factorial):
    matched.to_csv(output / "matched_runs.csv.gz", index=False)
    factorial.to_csv(output / "matched_factorial.csv", index=False)
    means = []
    for scope, block in shared_analysis.analysis_strata(matched):
        for arm, rows in block.groupby("arm"):
            for metric in ("end_separation", "end_excess"):
                means.append({**scope, "arm": arm, "metric": metric, **shared_analysis.mean_interval(rows[metric])})
    pd.DataFrame(means).to_csv(output / "matched_arm_means.csv", index=False)
    strata = ["sensitivity", "epsilon", "history", "arm"]
    matched.groupby([*strata, "outcome"]).size().reset_index(name="n").to_csv(
        output / "matched_outcome_counts.csv", index=False)
    matched.groupby([*strata, "setting", "influence", "rewiring", "repost", "pathway"]).size().reset_index(name="n").to_csv(
        output / "matched_pathway_counts.csv", index=False)
    if "OpinionM9" in set(matched["arm"]):
        paired = pd.concat([
            shared_analysis.paired_effects(matched, arm, "OpinionM9").assign(arm=arm, control="OpinionM9")
            for arm in sorted(set(matched["arm"]) - {"OpinionM9"})
        ], ignore_index=True)
        paired.to_csv(output / "paired_vs_m9.csv.gz", index=False)
        summaries = []
        for scope, block in shared_analysis.analysis_strata(paired):
            for arm, rows in block.groupby("arm"):
                for metric in ("delta_separation", "delta_excess"):
                    summaries.append({**scope, "arm": arm, "control": "OpinionM9", "metric": metric,
                                      **shared_analysis.mean_interval(rows[metric])})
        pd.DataFrame(summaries).to_csv(output / "paired_vs_m9_summary.csv", index=False)


def prepare(bundle, output):
    """Authenticate local observations and preserve matched units and estimates."""
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    manifest = shared_bundle._bundle_manifest(bundle)
    plan = read_plan(bundle / "data/study/plan.json")
    if plan["profile"] != "recommender_variants":
        raise ValueError("factorial review requires a recommender_variants bundle")
    runs = pd.read_csv(bundle / "data/tables/runs.csv.gz", float_precision="round_trip")
    matched = matched_factorial_runs(runs)
    if matched.empty:
        raise ValueError("no initial state has all four OpinionRandom configurations")
    configurations = matched[matched["recommender"] == "OpinionRandom"]
    arms = [configurations[(configurations["zeta"] == zeta) & (configurations["score_noise"] == noise)]
            ["arm"].iloc[0] for zeta, noise in FACTORS]
    labels = [f"ζ={zeta}\nσ={noise}" for zeta, noise in FACTORS]
    if "OpinionM9" in set(matched["arm"]):
        if set(matched.loc[matched["arm"] == "OpinionM9", "task_id"]) != set(matched["task_id"]):
            raise ValueError("M9 reference is missing from the matched factorial cohort")
        arms.append("OpinionM9")
        labels.append("Opinion\nM9")
    factorial = opinion_random_factorial(plan, matched)
    strata = []
    for (sensitivity, epsilon, history), block in matched.groupby(["sensitivity", "epsilon", "history"]):
        mask = ((factorial["sensitivity"] == sensitivity) & (factorial["epsilon"] == epsilon)
                & (factorial["history"] == history))
        observations = block[block["arm"].isin(arms)]
        planned = sum(task["sensitivity"] == sensitivity
                      and task["metadata"]["HKParams"]["Tolerance"] == epsilon
                      and task["metadata"]["PostRetainCount"] == history for task in plan["tasks"])
        strata.append({
            "name": f"{sensitivity}_e{round(100 * epsilon):02}_kh{int(history)}",
            "title": f"ε={epsilon:g}, history={int(history)}; {block['task_id'].nunique()}/{planned} matched tasks",
            "observations": _rows(observations[["task_id", "arm", "setting", "influence", "rewiring", "repost",
                                                "end_separation", "end_excess", "pathway"]]),
            "factorial": _rows(factorial[mask & factorial["metric"].isin(("separation_change", "excess_change"))]),
        })
    output.mkdir(parents=True, exist_ok=True)
    _write_matched_tables(output, matched, factorial)
    atomic_json(output / "plot_data.json", {
        "schema": SCHEMA, "plan_fingerprint": plan["plan_fingerprint"],
        "bundle_manifest_sha256": sha(bundle / "data/manifest.json"),
        "source_completed": manifest["completed"], "source_planned": len(plan["tasks"]),
        "arms": arms, "labels": labels, "strata": strata,
        "cohort": "All four OpinionRandom configurations completed for the same task; original M9 reference uses those tasks.",
    })
    if not (output / "layout.json").exists():
        atomic_json(output / "layout.json", {
            "schema": SCHEMA, "font_size": 9, "summary_size": [8.6, 6.8],
            "cell_size": [10.4, 7.3], "cell_limit": None, "cmap": "RdBu_r",
        })
    return {"matched": int(matched["task_id"].nunique()), "planned": len(plan["tasks"])}


def _summaries(data, scope, layout, destination):
    rows = pd.DataFrame(scope["observations"])
    fig, axes = plt.subplots(2, 2, figsize=layout["summary_size"], constrained_layout=True)
    fig.suptitle(scope["title"])
    for ax, metric, title in zip(axes[0], ("end_separation", "end_excess"),
                               ("(a) Endpoint separation", "(b) Endpoint excess concordance")):
        for index, arm in enumerate(data["arms"]):
            estimate = shared_analysis.mean_interval(rows.loc[rows["arm"] == arm, metric])
            color = "#444444" if arm == "OpinionM9" else "#2171b5"
            ax.plot(index, estimate["mean"], "o", color=color)
            if estimate["lo"] is not None:
                ax.vlines(index, estimate["lo"], estimate["hi"], color=color)
        ax.set(xticks=range(len(data["arms"])), xticklabels=data["labels"], title=title,
               ylabel="Mean P" if metric == "end_separation" else "Mean E")
        ax.axhline(0, color="0.7", lw=.7)
    ax = axes[1, 0]
    effects = pd.DataFrame(scope["factorial"])
    effects = effects[(effects["setting"] == "all") & (effects["repost"] == "all")
                      & (effects["metric"] == "separation_change")].set_index("contrast")
    ax.axvline(0, color="0.6", lw=.8)
    for index, contrast in enumerate(FACTORIAL_CONTRASTS):
        row = effects.loc[contrast]
        ax.plot(row["mean"], index, "o", color="#2171b5")
        if pd.notna(row["lo"]):
            ax.hlines(index, row["lo"], row["hi"], color="#2171b5")
    ax.set(yticks=range(5), yticklabels=CONTRAST_LABELS, ylim=(4.5, -.5),
           xlabel="Paired ΔP and 95% t interval", title="(c) Factor effects on endpoint P")
    ax = axes[1, 1]
    balanced = rows[np.isclose(rows["influence"], rows["rewiring"])]
    bottom = np.zeros(len(data["arms"]))
    for pathway, color in zip(PATHWAY_LABELS, PATHWAY_COLORS):
        counts = np.array([int(((balanced["arm"] == arm) & (balanced["pathway"] == pathway)).sum())
                           for arm in data["arms"]])
        if counts.any():
            ax.bar(range(len(bottom)), counts, bottom=bottom, color=color, label=pathway.replace("_", " "))
            bottom += counts
    ax.set(xticks=range(len(bottom)), xticklabels=data["labels"], ylabel="Number of tasks",
           title=f"(d) Pathways at α=q (n={balanced['task_id'].nunique()})")
    if balanced.empty:
        ax.text(.5, .5, "No matched α=q tasks", transform=ax.transAxes, ha="center")
    else:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ["Same step" if name == "simultaneous" else name for name in labels],
                  fontsize=7, loc="upper center", ncol=min(3, len(labels)), frameon=False)
        ax.set_ylim(0, max(bottom) * 1.25)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    shared_plot._save_panel(fig, destination / f"{scope['name']}_summary")


def _cells(scope, layout, destination):
    rows = pd.DataFrame(scope["observations"])
    settings = sorted(set(zip(rows["influence"], rows["rewiring"])))
    if len(settings) < 2:
        return
    reposts = sorted(rows["repost"].unique())
    effects = pd.DataFrame(scope["factorial"])
    effects = effects[(effects["setting"] != "all") & (effects["metric"] == "separation_change")]
    matrices = []
    counts = []
    for contrast in FACTORIAL_CONTRASTS:
        matrix = np.full((len(settings), len(reposts)), np.nan)
        count = np.zeros_like(matrix, dtype=int)
        for i, (influence, rewiring) in enumerate(settings):
            cells = rows[(rows["influence"] == influence) & (rows["rewiring"] == rewiring)]
            setting, = cells["setting"].unique()
            for j, repost in enumerate(reposts):
                ids = set(cells.loc[cells["repost"] == repost, "task_id"])
                sample = effects[(effects["contrast"] == contrast) & (effects["repost"] == repost)
                                 & (effects["setting"] == setting)]
                if len(ids):
                    if len(sample) != 1 or int(sample.iloc[0]["n"]) != len(ids):
                        raise ValueError("cell contrast does not use its matched factorial cohort")
                    matrix[i, j], count[i, j] = sample.iloc[0]["mean"], len(ids)
        matrices.append(matrix)
        counts.append(count)
    limit = layout["cell_limit"] or max([.001, *(float(np.nanmax(np.abs(m))) for m in matrices if np.isfinite(m).any())])
    fig, axes = plt.subplots(2, 3, figsize=layout["cell_size"], constrained_layout=True)
    available_counts = np.unique(np.concatenate([count[count > 0] for count in counts]))
    common_count = int(available_counts[0]) if len(available_counts) == 1 else None
    note = f"n={common_count} paired repeats per cell" if common_count else "cell labels show mean and n"
    fig.suptitle(scope["title"] + " | " + note)
    for index, (ax, matrix, count, title) in enumerate(zip(axes.flat, matrices, counts, CONTRAST_LABELS)):
        shown = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap=layout["cmap"], aspect="auto")
        ax.set(xticks=range(len(reposts)), xticklabels=[f"p={p:g}" for p in reposts],
               yticks=range(len(settings)), yticklabels=[f"{a:g}, {q:g}" for a, q in settings],
               ylabel="α, q", title=f"({chr(97 + index)}) {title}")
        for i, j in np.ndindex(matrix.shape):
            if np.isfinite(matrix[i, j]):
                text = f"{matrix[i,j]:+.3f}" + (f"\nn={count[i,j]}" if common_count is None else "")
                ax.text(j, i, text, ha="center", va="center", fontsize=layout.get("cell_annotation_size", 6),
                        color="white" if abs(matrix[i,j]) > .65 * limit else "black")
    axes[1, 2].set_axis_off()
    fig.colorbar(shown, cax=axes[1, 2].inset_axes([.3, .1, .08, .8]), label="Mean paired ΔP")
    shared_plot._save_panel(fig, destination / f"{scope['name']}_cells")


def compose(directory):
    """Render only saved plotting observations and the editable layout."""
    directory = Path(directory).resolve()
    data = json.loads((directory / "plot_data.json").read_text())
    layout = json.loads((directory / "layout.json").read_text())
    if data["schema"] != SCHEMA or layout["schema"] != SCHEMA:
        raise ValueError("unsupported factorial figure schema")
    scopes = {scope["name"]: scope for scope in data["strata"]}
    selected = layout.get("strata", list(scopes))
    if not selected or len(set(selected)) != len(selected) or set(selected) - scopes.keys():
        raise ValueError("layout must select distinct available strata")
    setup_paper_params()
    plt.rcParams.update({"font.size": layout["font_size"], "axes.titlesize": layout["font_size"],
                         "axes.labelsize": layout["font_size"], "xtick.labelsize": layout["font_size"],
                         "ytick.labelsize": layout["font_size"], "figure.titlesize": layout["font_size"] + 1,
                         "svg.fonttype": "none"})
    destination = directory / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    for name in selected:
        scope = scopes[name]
        _summaries(data, scope, layout, destination)
        _cells(scope, layout, destination)
    rendered = {name + suffix for name in selected for suffix in ("_summary", "_cells")}
    atomic_json(directory / "figure_manifest.json", {
        "plot_data_sha256": sha(directory / "plot_data.json"), "layout_sha256": sha(directory / "layout.json"),
        "renderer_sha256": sha(Path(__file__)),
        "strata": selected,
        "files": {p.name: sha(p) for p in sorted(destination.iterdir())
                  if p.stem in rendered and p.suffix in (".png", ".pdf", ".svg")},
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--bundle", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    compose_parser = commands.add_parser("compose")
    compose_parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.bundle, args.output)))
    else:
        compose(args.directory)


if __name__ == "__main__":
    main()
