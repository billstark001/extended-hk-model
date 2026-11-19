"""Selected manuscript layouts from frozen CSV tables, with no model execution.

This file is kept byte-identical to the model repository's evidence module.
"""

from __future__ import annotations

import csv
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter

COLORS = ("#555555", "#86b5d8", "#166da8", "#eba981", "#bd4e19")
LABELS = ("Random", "Opinion1", "Opinion4", "Structure1", "Structure4")
KERNELS = (
    ("random", 1),
    ("opinion_random", 1),
    ("opinion_random", 4),
    ("structure_random_l0", 1),
    ("structure_random_l0", 4),
)
RULES = ("hk", "deffuant")
EPS = (0.2, 0.4, 0.8)


def select(rows, **values):
    return [row for row in rows if all(row[key] == value for key, value in values.items())]


def one(rows, **values):
    hits = select(rows, **values)
    if len(hits) != 1:
        raise ValueError(f"expected one selected row, found {len(hits)}: {values}")
    return hits[0]


def decorate(axes, labels=None):
    for idx, ax in enumerate(np.asarray(axes, dtype=object).flat):
        title = ax.get_title()
        ax.set_title("")
        label = labels[idx] if labels is not None else chr(97 + idx)
        ax.set_title(f"({label}) {title}".rstrip(), loc="left", pad=4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.14, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.yaxis.get_offset_text().set_x(0.15)
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(NullFormatter())


def save(fig, output, name):
    fig.savefig(output / (name + ".pdf"), dpi=300)
    fig.savefig(output / (name + ".png"), dpi=220)
    plt.close(fig)


def write_rows(output, name, rows):
    with (output / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def conditions():
    return [
        Line2D([], [], color=COLORS[idx], ls=ls, label=LABELS[idx] + " / " + label)
        for idx in (0, 2)
        for ls, label in (("-", "HK"), ("--", "Deffuant"))
    ]


def response_factors(tables, output):
    rows = select(
        tables["conditional_means.csv"],
        population=500,
        bins=161,
        quadrature=33,
        horizon=480,
        plateau_reference="own_endpoint",
        processes=8,
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.25), layout="constrained")
    derived, errors = [], []
    for ri, rule in enumerate(RULES):
        for initial, marker in ((0, "o"), (1, "s")):
            for epsilon in EPS:
                common = dict(epsilon=epsilon, dynamics=rule, initial=initial)
                baseline = one(rows, **common, recommender="random", steepness=1)
                opinion = one(rows, **common, recommender="opinion_random", steepness=4)
                x = epsilon + (ri - 0.5) * 0.015 + (initial - 0.5) * 0.006
                for ax, channel in zip(axes[:2], ("node", "edge")):
                    value = opinion[channel + "_mean"] / baseline[channel + "_mean"]
                    prediction = opinion["pair_" + channel + "_mean"] / baseline["pair_" + channel + "_mean"]
                    color = (COLORS[2], COLORS[4])[ri]
                    ax.plot(x, value, marker, color=color, ms=3.2)
                    ax.plot(x + 0.005, prediction, "x", color=color, ms=3.2)
                    derived.append(
                        {
                            **common,
                            "channel": channel,
                            "microscopic_factor": value,
                            "map_factor": prediction,
                            "opinion_area": opinion[channel + "_mean"],
                            "random_area": baseline[channel + "_mean"],
                            "opinion_map_area": opinion["pair_" + channel + "_mean"],
                            "random_map_area": baseline["pair_" + channel + "_mean"],
                        }
                    )
                for idx in (0, 2, 4):
                    rec, exponent = KERNELS[idx]
                    item = one(rows, **common, recommender=rec, steepness=exponent)
                    for channel in ("node", "edge"):
                        errors.append(
                            {
                                **common,
                                "recommender": rec,
                                "steepness": exponent,
                                "channel": channel,
                                "absolute_relative_error": abs(
                                    item["pair_" + channel + "_mean"] / item[channel + "_mean"] - 1
                                ),
                            }
                        )
    for ax, channel in zip(axes[:2], ("Node", "Edge")):
        ax.axhline(1, color=".45", ls=":", lw=0.7)
        ax.set(title=f"{channel} contrast", ylabel="Opinion4 / Random area")
    ranges: dict[str, dict[str, list[float]]] = {}
    for ax, channel in zip(axes[2:], ("node", "edge")):
        ranges[channel] = {}
        for epsilon in EPS:
            values = [
                100 * row["absolute_relative_error"]
                for row in errors
                if row["epsilon"] == epsilon and row["channel"] == channel
            ]
            if len(values) != 12:
                raise ValueError("area-error ranges require twelve matched configurations")
            lo, hi = min(values), max(values)
            ax.vlines(epsilon, lo, hi, color=".25", lw=1.1)
            ax.plot([epsilon, epsilon], [lo, hi], "_", color=".25", ms=7)
            ax.plot(epsilon + np.linspace(-0.017, 0.017, len(values)), values, ".", color=".65", ms=2.1)
            ranges[channel][f"{epsilon:g}"] = [lo, hi]
        ax.set(title=f"{channel.capitalize()} area error", ylabel="Relative area error (%)", ylim=(0, None))
    for ax in axes:
        ax.set(xlabel=r"Confidence radius $\epsilon$", xticks=EPS, xticklabels=(".2", ".4", ".8"), xlim=(0.15, 0.85))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    decorate(axes)
    handles = [Line2D([], [], color=c, label=rule) for c, rule in ((COLORS[2], "HK"), (COLORS[4], "Deffuant"))]
    handles += [
        Line2D([], [], color=".3", marker=m, ls="none", label=label)
        for m, label in (("o", "Initial 0"), ("s", "Initial 1"), ("x", "Measure map"))
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    save(fig, output, "supp_response_factors")
    write_rows(output, "response_factors.csv", derived)
    write_rows(output, "area_prediction_errors.csv", errors)
    ratio_ranges = {
        f"{epsilon:g}": {
            channel: {
                source: [
                    min(
                        row[source + "_factor"]
                        for row in derived
                        if row["epsilon"] == epsilon and row["channel"] == channel
                    ),
                    max(
                        row[source + "_factor"]
                        for row in derived
                        if row["epsilon"] == epsilon and row["channel"] == channel
                    ),
                ]
                for source in ("microscopic", "map")
            }
            for channel in ("node", "edge")
        }
        for epsilon in EPS
    }
    return {
        "selected_conditional_rows": len(rows),
        "ratio_rows": len(derived),
        "error_rows": len(errors),
        "ratio_ranges": ratio_ranges,
        "area_error_ranges_percent": ranges,
        "uncertainty": "ratio point estimates only; error bars are min-max over twelve point estimates, not confidence intervals",
        "selection": "N=500; tau<=480; eight processes per initial state; own_endpoint; two initial states",
        "caption": "Matched recommendation response factors (a,b), and absolute area-prediction errors relative to microscopic values (c,d), across confidence radii. Error bars span twelve point estimates, not uncertainty intervals.",
    }


def selected_outcomes(tables):
    terminal = select(tables["population_outcomes.csv"], epsilon=0.4, tau=120)
    if len(terminal) != 256:
        raise ValueError("terminal outcome selection must contain all 256 central-confidence processes")
    keys = ("dynamics", "recommender", "steepness", "population", "initial")
    near = [row for row in terminal if row["near_consensus_1e8"] == "True"]
    groups = sorted(
        {tuple(row[key] for key in keys) for row in near}, key=lambda group: (group[1] != "random", group[0] != "hk")
    )
    if len(near) != 6 or len(groups) != 3:
        raise ValueError("expected all six near-consensus outcomes in three initial-state groups")
    outcomes = []
    for group in groups:
        common = dict(zip(keys, group))
        processes = sorted(select(terminal, **common), key=lambda row: row["branch"])
        if len(processes) != 4:
            raise ValueError("each selected initial state must retain all four processes")
        for item in processes:
            prediction = one(
                tables["micro_branches.csv"],
                **common,
                branch=item["branch"],
                epsilon=0.4,
                ratio=2,
                bins=161,
                quadrature=33,
                horizon_tau=120,
            )
            if not np.isclose(item["variance"], prediction["micro_variance"], rtol=1e-12, atol=1e-30):
                raise ValueError("terminal outcome and matched-branch variance disagree")
            outcomes.append(
                {
                    **common,
                    "branch": item["branch"],
                    "micro_variance": item["variance"],
                    "map_variance": prediction["pair_variance"],
                    "near_consensus": item["near_consensus_1e8"],
                }
            )
    return groups, outcomes


def structure(tables, output):
    rows = select(tables["population_structure.csv"], epsilon=0.2, tau=120)
    groups, outcomes = selected_outcomes(tables)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), width_ratios=(1, 1, 1.12), layout="constrained")
    for ax, metric, title, ylabel, boundary in zip(
        axes[:2],
        ("opinion_variance", "excess_concordance"),
        ("Opinion differentiation", "Preferential pairing"),
        ("Mean opinion variance", "Mean excess concordance"),
        (1e-4, 0),
    ):
        for idx in (0, 2):
            rec, exponent = KERNELS[idx]
            for rule, ls in zip(RULES, ("-", "--")):
                rs = sorted(
                    select(rows, metric=metric, dynamics=rule, recommender=rec, steepness=exponent),
                    key=lambda row: row["population"],
                )
                ns = [row["population"] for row in rs]
                ax.fill_between(
                    ns, [row["lower"] for row in rs], [row["upper"] for row in rs], color=COLORS[idx], alpha=0.10
                )
                ax.plot(
                    ns,
                    [row["mean"] - row["signed_bias"] for row in rs],
                    "s" + ls,
                    color=COLORS[idx],
                    mfc="white",
                    ms=3.8,
                    lw=0.7,
                )
                ax.plot(ns, [row["mean"] for row in rs], "o" + ls, color=COLORS[idx], ms=2.8, zorder=3)
        ax.axhline(boundary, color=".4", ls=":", lw=0.7)
        ax.set(
            title=title,
            ylabel=ylabel,
            xlabel=r"Population $N$",
            xscale="log",
            xticks=(250, 500, 1000, 2000),
            xticklabels=("250", "500", "1000", "2000"),
        )
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_ylim(bottom=min(-0.01, ax.get_ylim()[0]))
    labels = []
    keys = ("dynamics", "recommender", "steepness", "population", "initial")
    for j, group in enumerate(groups):
        common = dict(zip(keys, group))
        rs = sorted(select(outcomes, **common), key=lambda row: row["branch"])
        color = COLORS[0 if common["recommender"] == "random" else 2]
        offsets = np.linspace(-0.21, 0.09, 4)
        for offset, item in zip(offsets, rs):
            axes[2].plot(j + offset, item["micro_variance"], "o", color=color, ms=3.1)
            axes[2].annotate(
                f"{int(item['branch'])}",
                (j + offset, item["micro_variance"]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        predictions = {item["map_variance"] for item in rs}
        if len(predictions) != 1:
            raise ValueError("repeated processes from one initial state must share one prediction")
        axes[2].plot(j + 0.23, predictions.pop(), "s", mfc="white", color=color, ms=4.3)
        rule = "HK" if common["dynamics"] == "hk" else "Deffuant"
        law = "Random" if common["recommender"] == "random" else "Opinion4"
        labels.append(f"{rule}\n{law}\n$N={common['population']:g}$")
    axes[2].axhline(0, color=".4", ls=":", lw=0.7)
    axes[2].set(
        title="Same initial state",
        ylabel="Terminal opinion variance",
        xticks=range(3),
        xticklabels=labels,
        xlim=(-0.4, 2.5),
        ylim=(-0.008, 0.20),
    )
    axes[2].yaxis.set_major_locator(MaxNLocator(4))
    decorate(axes)
    handles = conditions() + [
        Line2D([], [], color=".3", ls="none", marker=m, mfc=fill, label=label)
        for m, fill, label in (("o", ".3", "Microscopic"), ("s", "white", "Measure map"))
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    save(fig, output, "supp_structure")
    write_rows(output, "selected_process_outcomes.csv", outcomes)
    return {
        "structure_selection": "epsilon=0.2; r=2; tau=120; all four population sizes",
        "outcome_selection": "all three epsilon=.4 initial-state groups containing any near-consensus process at tau=120; all four processes per selected state; initial index 3 in each group",
        "selected_processes": len(outcomes),
        "near_consensus_processes": sum(row["near_consensus"] == "True" for row in outcomes),
        "uncertainty": "existing exploratory 95% whole-initial-group bootstrap intervals in a,b; raw process values in c",
        "caption": "Collective structure at epsilon=.2 (a,b), and all twelve processes in the three epsilon=.4 initial-state groups containing near-consensus outcomes (c). Numbers in c are process indices; each group's hollow square is its matched prediction.",
    }


def variability(tables, output):
    rows = select(tables["population_variation.csv"], population=2000, tau=120)
    components = ("process_variance", "initial_variance_corrected", "conditional_bias_squared_corrected")
    labels = (r"Process variance $W$", r"Initial-state variation $I$", r"Residual $\widehat{\mathrm{Err}}_{\rm mean}$")
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.45), sharex=True, layout="constrained")
    for row, metric, title in zip(axes, ("node_cdf", "edge_cdf"), ("Node", "Joint edge")):
        for ax, component, label, name in zip(row, components, labels, ("process", "initial", "prediction")):
            extent: list[float] = []
            for idx in (0, 2):
                rec, exponent = KERNELS[idx]
                for rule, ls in zip(RULES, ("-", "--")):
                    rs = sorted(
                        select(rows, metric=metric, dynamics=rule, recommender=rec, steepness=exponent),
                        key=lambda item: item["epsilon"],
                    )
                    xs = [item["epsilon"] for item in rs]
                    ax.plot(xs, [item[component] for item in rs], "o" + ls, color=COLORS[idx])
                    ax.fill_between(
                        xs,
                        [item[component + "_loio_min"] for item in rs],
                        [item[component + "_loio_max"] for item in rs],
                        color=COLORS[idx],
                        alpha=0.10,
                    )
                    extent.extend(item[component + suffix] for item in rs for suffix in ("", "_loio_min", "_loio_max"))
            ax.axhline(0, color=".5", lw=0.6)
            ax.set(
                title=title + " / " + name, ylabel=label, xticks=EPS, xticklabels=(".2", ".4", ".8"), xlim=(0.16, 0.84)
            )
            ax.set_yscale("symlog", linthresh=1e-4)
            ax.set_ylim(min(0, min(extent)) * 1.2 - 3e-6, max(extent) * 1.25)
    for ax in axes[-1]:
        ax.set_xlabel(r"Confidence radius $\epsilon$")
    decorate(axes)
    fig.legend(handles=conditions(), loc="outside lower center", ncol=4, frameon=False)
    save(fig, output, "supp_variability")
    selected = [row for row in rows if row["metric"] in ("node_cdf", "edge_cdf")]
    return {
        "selection": "N=2000; r=2; tau=120; all three confidence radii",
        "selected_rows": len(selected),
        "components": list(components),
        "uncertainty": "existing leave-one-initial-out extrema; not confidence intervals",
        "signed_estimates": "All raw signed estimates and LOIO bounds retained; symlog linthresh=1e-4",
        "interpretation": "I measures initial-state variation separately; matched prediction MSE=W+Err_mean excludes I",
        "caption": "Node and joint-edge CDF process variation W, corrected initial-state variation I, and corrected conditional-mean prediction residual at N=2000. Shading is LOIO sensitivity, not a confidence interval; signed estimates are retained. I is separate from the matched prediction-error identity.",
    }


def matched_compact(tables, output):
    rows = select(
        tables["matched_curves.csv"], epsilon=0.4, population=500, bins=161, quadrature=33, horizon=480, processes=8
    )
    areas = select(
        tables["conditional_means.csv"],
        epsilon=0.4,
        population=500,
        bins=161,
        quadrature=33,
        horizon=480,
        plateau_reference="own_endpoint",
        processes=8,
    )
    fig = plt.figure(figsize=(7.1, 4.7), layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=(1, 1, 0.95))
    top: list[Axes] = []
    for j, (channel, rule, initial) in enumerate((c, r, i) for c in ("node", "edge") for r in RULES for i in (0, 1)):
        row, col = divmod(j, 4)
        ax = fig.add_subplot(grid[row, col], sharey=top[j - 1] if col % 2 else None, sharex=top[col] if row else None)
        top.append(ax)
        for idx in (0, 2, 4):
            rec, exponent = KERNELS[idx]
            rs = sorted(
                select(rows, dynamics=rule, initial=initial, channel=channel, recommender=rec, steepness=exponent),
                key=lambda item: item["tau"],
            )
            xs = [item["tau"] for item in rs]
            ax.fill_between(
                xs,
                [item["lower"] for item in rs],
                [item["upper"] for item in rs],
                color=COLORS[idx],
                alpha=0.16,
                lw=0,
                rasterized=True,
            )
            ax.plot(xs, [item["mean"] for item in rs], color=COLORS[idx], ls="-")
            ax.plot(xs, [item["pair"] for item in rs], color=COLORS[idx], ls="--")
        symbol = r"\rho" if channel == "node" else "E"
        ax.set_ylabel(rf"$\mathcal{{U}}_{symbol}/\mathcal{{E}}_\star$")
        ax.set_title(("HK" if rule == "hk" else "Deffuant") + f" / {initial}" if row == 0 else "")
        ax.set_xscale("symlog", linthresh=1)
        ax.set(xlim=(0, 480), xticks=(0, 1, 10, 480), xticklabels=("0", "1", "10", "480"))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        if col % 2:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        if row == 0:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel(r"Scaled time $\tau$")
    bottom = [fig.add_subplot(grid[-1, :2]), fig.add_subplot(grid[-1, 2:])]
    for ax, channel in zip(bottom, ("node", "edge")):
        for j, idx in enumerate((0, 2, 4)):
            for rule, offset in zip(RULES, (-0.16, 0.16)):
                for initial, marker in ((0, "o"), (1, "s")):
                    item = one(
                        areas, dynamics=rule, initial=initial, recommender=KERNELS[idx][0], steepness=KERNELS[idx][1]
                    )
                    x = j + offset + (initial - 0.5) * 0.1
                    ax.vlines(x, item[channel + "_lower"], item[channel + "_upper"], color=COLORS[idx], lw=0.7)
                    ax.plot(
                        x,
                        item[channel + "_mean"],
                        marker,
                        mec=COLORS[idx],
                        mfc=COLORS[idx] if rule == "hk" else "white",
                        ms=3.2,
                    )
                    ax.plot(x + 0.035, item["pair_" + channel + "_mean"], "x", color=COLORS[idx], ms=3.6)
        symbol = r"\rho" if channel == "node" else "E"
        ax.set(
            xticks=range(3),
            xticklabels=[LABELS[i] for i in (0, 2, 4)],
            ylabel=rf"$\widehat\Theta_{symbol}$",
            title=channel.capitalize() + " area",
            xlim=(-0.45, 2.45),
            yscale="log",
        )
    decorate([*top, *bottom], [f"{chr(97 + j % 4)}{j // 4 + 1}" for j in range(8)] + ["e", "f"])
    handles = [Line2D([], [], color=COLORS[i], label=LABELS[i]) for i in (0, 2, 4)]
    handles += [
        Line2D([], [], color=".3", ls=ls, label=label) for ls, label in (("-", "Microscopic"), ("--", "Measure map"))
    ]
    handles += [
        Line2D([], [], color=".3", ls="none", marker=marker, label=label)
        for marker, label in (("o", "Initial 0"), ("s", "Initial 1"), ("x", "Measure map area"))
    ]
    handles += [
        Line2D([], [], color=".3", ls="none", marker="o", mfc=fill, label=label)
        for fill, label in ((".3", "HK (filled)"), ("white", "Deffuant (hollow)"))
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    save(fig, output, "figure_2_matched_v1")
    return {
        "selection": "epsilon=.4; N=500; r=2; tau<=480; two rules; two initial states; eight processes per state",
        "matched_curve_rows": len(rows),
        "conditional_area_rows": len(areas),
        "layout": "same eight energy panels and two area panels, 7.1x4.7 inches; existing 8pt axes/labels retained",
        "uncertainty": "original whole-process bootstrap mean-energy bands and conditional area intervals, unchanged",
        "caption": "Raw microscopic and matched measure-map node/edge energy histories, with original conditional bootstrap bands. Columns are HK/0, HK/1, Deffuant/0, Deffuant/1. Bottom panels retain all own-reference areas, intervals and map predictions.",
    }


def render(tables, output, *, include=None):
    """Render every selected layout whose validated tables are supplied."""
    settings: dict[Any, Any] = dict(plt.rcParamsDefault)
    settings.pop("backend", None)
    settings.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.85,
            "lines.markersize": 3.0,
            "lines.markeredgewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    figures = {}
    with plt.rc_context(settings):
        if "conditional_means.csv" in tables and (include is None or "supp_response_factors" in include):
            figures["supp_response_factors"] = response_factors(tables, output)
        if all(
            name in tables for name in ("population_structure.csv", "population_outcomes.csv", "micro_branches.csv")
        ) and (include is None or "supp_structure" in include):
            figures["supp_structure"] = structure(tables, output)
        if "population_variation.csv" in tables and (include is None or "supp_variability" in include):
            figures["supp_variability"] = variability(tables, output)
        if all(name in tables for name in ("conditional_means.csv", "matched_curves.csv")) and (
            include is None or "figure_2_matched_v1" in include
        ):
            figures["figure_2_matched_v1"] = matched_compact(tables, output)
    return figures
