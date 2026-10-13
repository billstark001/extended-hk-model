"""Current main figures assembled exclusively from this run's frozen records."""

import json
from collections import defaultdict
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common.io.json import atomic_json
from common.io.records import record_path
from common.plotting.figures import decorate
from common.statistics import bootstrap_weights
from theory.runtime.records import read_micro
from theory.validation import fields_on

from ..runtime.measure_analysis import cached_cases
from ..types import FrozenMainPlan
from .figure_style import COLORS, RULES, clock_label, color_legend, matches, read_table, save
from .protocol import KERNELS


def mean_figure(directory, output, plan, cases):
    from .table_figures import absolute_figure, recommendation_figure

    recommendation_figure(directory, output, plan)
    absolute_figure(directory, output, plan)


def matched_records(directory, plan):
    for task in plan["tasks"]:
        if task["kind"] == "micro" and "matched" in task["roles"]:
            record = read_micro(record_path(directory / "micro" / "raw", task["id"]))
            for frames in (record["result"]["frames"], record["result"]["pair"]["frames"]):
                for frame in frames:
                    frame.pop("rho", None)
                    frame.pop("edge", None)
            yield record


def micro_figure(directory, output, plan):
    protocol = plan["analysis_protocol"]
    groups = defaultdict(list)
    for record in matched_records(directory, plan):
        task, p = record["task"], record["task"]["parameters"]
        groups[(p["dynamics"], p["recommender"], p["steepness"], task["initial"])].append(record)
    rows = [
        r
        for r in read_table(directory, "conditional_means", protocol["baseline_epsilon"])
        if matches(
            r,
            population=protocol["matched_population"],
            bins=protocol["matched_bins"],
            quadrature=protocol["matched_quadrature"],
            plateau_reference="own_endpoint",
            horizon=protocol["maximum_horizon_tau"],
        )
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    selected = (("random", 1.0, COLORS[0]), ("opinion_random", 4.0, COLORS[2]), ("structure_random_l0", 4.0, COLORS[4]))
    for col, rule in enumerate(RULES):
        for rec, z, color in selected:
            for initial in (0, 1):
                members = groups[(rule, rec, z, initial)]
                first = members[0]["result"]
                time = np.array([f["time"] for f in first["frames"]]) * protocol["baseline_alpha"]
                pair = first["pair"]["frames"]
                pt = np.array([f["time"] for f in pair]) * protocol["baseline_alpha"]
                weights = bootstrap_weights(len(members), protocol)
                for i, channel in enumerate(("node", "edge")):
                    values = np.array(
                        [[f["exact_" + channel + "_energy"] for f in m["result"]["frames"]] for m in members]
                    )
                    low, high = np.quantile(weights @ values, [0.025, 0.975], axis=0)
                    ax = axes[i, col]
                    ax.fill_between(time, low, high, color=color, alpha=0.10)
                    ax.plot(time, values.mean(0), color=color, ls="-" if initial == 0 else ":", lw=1.6)
                    ax.plot(
                        pt,
                        [f[channel + "_energy"] for f in pair],
                        color=color,
                        ls="--" if initial == 0 else "-.",
                        lw=1.1,
                    )
        for i, channel in enumerate(("node", "edge")):
            axes[i, col].set(xlabel=r"$\tau=\alpha t$", ylabel=f"Raw {channel} energy")
        axes[0, col].set_title("HK" if rule == "hk" else "One-sided Deffuant")
        ax = axes[2, col]
        for j, (rec, z, color) in enumerate(selected):
            for initial in (0, 1):
                row = next(r for r in rows if matches(r, dynamics=rule, recommender=rec, steepness=z, initial=initial))
                for i, channel in enumerate(("node", "edge")):
                    x = j + (i - 0.5) * 0.32 + (initial - 0.5) * 0.10
                    center = row[channel + "_mean"]
                    low, high = row[channel + "_lower"], row[channel + "_upper"]
                    # Percentile bounds need not contain the point estimate.
                    ax.vlines(x, low, high, color=color, alpha=0.7)
                    ax.plot(x, center, "o" if initial == 0 else "s", color=color, ms=4)
                    ax.plot(x, row["pair_" + channel + "_mean"], "x", color=color, ms=5)
        ax.set(
            xticks=range(3),
            xticklabels=["Random", "Opinion4", "Structure4"],
            ylabel="Node / edge residual areas (left / right)",
        )
    decorate(axes)
    handles = [
        Line2D([], [], color=c, label=label)
        for c, label in zip((COLORS[0], COLORS[2], COLORS[4]), ("Random", "Opinion4", "Structure4"))
    ]
    handles += [
        Line2D([], [], color="k", ls=ls, label=label)
        for ls, label in (
            ("-", "Micro initial 0"),
            (":", "Micro initial 1"),
            ("--", "Pair initial 0"),
            ("-.", "Pair initial 1"),
        )
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    horizon = max(protocol["prefix_taus"])
    fig.suptitle(
        "Matched initial graphs; whole-process 95% bootstrap bands\n"
        + clock_label(horizon)
        + "; circles/squares: initial 0/1; crosses: pair",
        fontsize=11,
    )
    save(fig, output, "mean_micro_validation", protocol["baseline_epsilon"])


def population_figure(directory, output, plan):
    protocol = plan["analysis_protocol"]
    kernels = [tuple(k) for k in protocol.get("population_kernels", KERNELS)]
    rows = [
        r
        for r in read_table(directory, "population", protocol["baseline_epsilon"])
        if matches(
            r,
            bins=protocol["main_population_bins"],
            quadrature=protocol["main_population_quadrature"],
            horizon=protocol["population_horizon_tau"],
        )
    ]
    tasks = [
        t
        for t in plan["tasks"]
        if t["kind"] == "micro"
        and "population" in t["roles"]
        and matches(
            t["parameters"],
            dynamics="hk",
            recommender="opinion_random",
            steepness=4.0,
            population=protocol["matched_population"],
        )
        and t["initial"] == 0
    ]
    pairs, micros = [], []
    for task in tasks:
        result = read_micro(record_path(directory / "micro" / "raw", task["id"]))["result"]

        def snapshot(frames):
            return next(
                f
                for f in frames
                if np.asarray(f.get("edge", [])).size
                and np.isclose(f["time"] * protocol["baseline_alpha"], protocol["main_snapshot_tau"])
            )

        micros.append(fields_on(snapshot(result["frames"]), protocol["main_population_bins"]))
        pairs.append(fields_on(snapshot(result["pair"]["frames"]), protocol["main_population_bins"]))
    rho, edge = (np.mean([m[i] for m in micros], axis=0) for i in (0, 1))
    p, e = pairs[0]
    bins = len(p)
    x = -1 + (np.arange(bins) + 0.5) * 2 / bins
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    axes[0, 0].plot(x, p * bins / 2, color=COLORS[2], label="Pair")
    axes[0, 0].plot(x, rho * bins / 2, color="k", ls="--", label="Micro branch mean")
    axes[0, 0].set(xlabel="Opinion", ylabel=r"Node density $\rho$", title="HK Opinion4, initial 0")
    axes[0, 0].legend(frameon=False)
    vmax = max(np.max(e / e.sum()), np.max(edge / edge.sum())) * bins**2 / 4
    for ax, matrix, title in (
        (axes[0, 1], e, "Pair directed-edge density"),
        (axes[0, 2], edge, "Micro directed-edge density"),
    ):
        heat = ax.imshow(
            (matrix / matrix.sum()).T * bins**2 / 4,
            origin="lower",
            extent=(-1, 1, -1, 1),
            vmin=0,
            vmax=vmax,
            cmap="magma",
            aspect="equal",
        )
        ax.set(xlabel="Source opinion", ylabel="Target opinion", title=title)
    fig.colorbar(heat, ax=axes[0, 1:], shrink=0.75, label=r"$E/\bar k$ density")
    metrics = (
        ("micro_variance", "Population variance"),
        ("micro_pairing_excess", "Binned concordance minus marginal reference"),
        ("node_w1", r"Micro--pair node $W_1$"),
        ("node", "Node residual area"),
        ("edge", "Edge residual area"),
        ("edge_to_node", "Edge / node area ratio"),
    )
    for ax, (metric, label) in zip(axes[1:].flat, metrics):
        for color, (rec, z) in zip(COLORS, KERNELS):
            if (rec, z) not in kernels:
                continue
            for rule, ls in (("hk", "-"), ("deffuant", "--")):
                group = sorted(
                    (r for r in rows if matches(r, dynamics=rule, recommender=rec, steepness=z)),
                    key=lambda r: r["population"],
                )
                xs = [r["population"] for r in group]
                ys = [r[metric + "_mean"] for r in group]
                ax.plot(xs, ys, "o" + ls, color=color, ms=3)
                ax.fill_between(
                    xs,
                    [r[metric + "_lower"] for r in group],
                    [r[metric + "_upper"] for r in group],
                    color=color,
                    alpha=0.09,
                )
        ax.set(xlabel="Population N", ylabel=label)
        ax.set_xticks(sorted({r["population"] for r in rows}))
    decorate(axes)
    fig.legend(
        handles=color_legend(kernels)
        + [Line2D([], [], color="k", ls="-", label="HK"), Line2D([], [], color="k", ls="--", label="Deffuant")],
        loc="outside lower center",
        ncol=7,
        frameon=False,
    )
    fig.suptitle(
        f"Snapshots at tau={protocol['main_snapshot_tau']:g}; fixed k=15, R=10, r=2\n"
        + clock_label(protocol.get("population_horizon_tau", max(protocol["prefix_taus"])))
        + "; hierarchical 95% bootstrap intervals",
        fontsize=11,
    )
    save(fig, output, "population_robustness", protocol["baseline_epsilon"])


def supplementary_figures(directory, output, plan, cases):
    tasks = {t["id"]: t for t in plan["tasks"]}
    selected = [
        c
        for c in cases
        if "scaling" in tasks[c["source"].removesuffix(".npz")]["roles"]
        and c["recommender"] == "opinion_random"
        and c["steepness"] == 4
        and c["ratio"] == 2
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    alphas = sorted({c["alpha"] for c in selected}, reverse=True)
    for col, rule in enumerate(RULES):
        for row, channel in enumerate(("node", "edge")):
            for alpha in alphas:
                c = next(c for c in selected if c["dynamics"] == rule and c["alpha"] == alpha)
                axes[row, col].plot(c["tau"], c[channel + "_energy"], label=f"alpha={alpha:g}")
            axes[row, col].set(xlabel=r"$\tau=\alpha t$", ylabel=f"Raw {channel} energy")
            if row == 0:
                axes[row, col].set_title("HK" if rule == "hk" else "One-sided Deffuant")
            if col == row == 0:
                axes[row, col].legend(frameon=False, fontsize=8)
    decorate(axes)
    fig.suptitle("Fixed r=2, Opinion4; common window/grid, no fitted shifts or endpoint alignment")
    save(fig, output, "scaling_collapse", plan["analysis_protocol"]["baseline_epsilon"])
    rows = read_table(directory, "numerics", plan["analysis_protocol"]["baseline_epsilon"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, metric, label in zip(
        axes,
        ("relative_max", "mean_relative_change"),
        ("Maximum raw-energy gap / baseline amplitude", "Residual-area relative change"),
    ):
        labels = [
            f"{r['dynamics'][:2]} {r['recommender'].split('_')[0]} {r['channel'][0]}\n"
            f"B{r['bins']:g} Q{r['quadrature']:g} dt{r['dt']:g}"
            for r in rows
        ]
        ax.scatter(range(len(rows)), [r[metric] for r in rows], s=12)
        ax.set(xticks=range(len(rows)), xticklabels=labels, ylabel=label)
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    decorate(axes)
    save(fig, output, "numerical_controls", plan["analysis_protocol"]["baseline_epsilon"])


def plot(directory):
    """Refuse partial coverage; generate each main and supplemental figure anew."""
    plan = cast(FrozenMainPlan, json.loads((directory / "plan.json").read_text()))
    report = json.loads((directory / "report.json").read_text())
    if report["missing"] or report["completed"] != report["planned"]:
        raise ValueError("main figures require the complete declared design")
    output = directory / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "savefig.facecolor": "white"})
    cases = list(cached_cases(directory / "measure"))
    confidence_radii = plan["analysis_protocol"].get(
        "confidence_radii", [plan["analysis_protocol"]["baseline_epsilon"]]
    )
    views = {}
    for epsilon in confidence_radii:
        # Each confidence radius has separate trajectories, ensembles and figure panels.
        view = dict(
            plan,
            tasks=[t for t in plan["tasks"] if t["parameters"]["epsilon"] == epsilon],
            analysis_protocol=dict(plan["analysis_protocol"], baseline_epsilon=epsilon),
        )
        destination = (
            output if epsilon == plan["analysis_protocol"]["baseline_epsilon"] else output / f"epsilon_{epsilon:g}"
        )
        destination.mkdir(exist_ok=True)
        selected = [c for c in cases if c["epsilon"] == epsilon]
        mean_figure(directory, destination, view, selected)
        micro_figure(directory, destination, view)
        population_figure(directory, destination, view)
        supplementary_figures(directory, destination, view, selected)
        views[str(epsilon)] = str(destination.relative_to(output))
    atomic_json(
        directory / "figure_manifest.json",
        {
            "plan_fingerprint": plan["plan_fingerprint"],
            "profile": plan["profile"],
            "main_figures": {
                "former_fig2_now_fig1": "mean_recommendation_clocks",
                "former_fig3_now_fig2": "mean_micro_validation",
                "former_fig4_now_fig3": "population_robustness",
            },
            "supplementary": ["absolute_release_areas", "scaling_collapse", "numerical_controls"],
            "baseline_epsilon": plan["analysis_protocol"]["baseline_epsilon"],
            "epsilon_directories": views,
            "clock_scope": "finite-window, own endpoint reference; tail confirmation is a separate diagnostic",
            "smoke_is_scientific_evidence": False,
            "input_scope": "only this run's plan, records and tables",
        },
    )
    print(f"generated {6 * len(confidence_radii)} PDF/PNG figure pairs in {output}", flush=True)
