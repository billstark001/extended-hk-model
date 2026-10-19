"""One rendering implementation for the revision figures, exclusively from local CSV."""

import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx  # type: ignore[import-untyped]
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import PowerNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, NullFormatter

from common.io.json import atomic_json, sha
from common.plotting import setup_paper_params
from common.plotting.figures import save
from common.runtime.failures import ProtocolError
from ehk.plotting import plot_network_snapshot

from .bundle import Bundle, csv_rows
from .design import is_smoke, topics
from .figure_style import COLORS, LABELS, RULES, matches
from .protocol import KERNELS


class Analysis:
    def __init__(self, directory):
        self.bundle = Bundle(directory)
        self.root, self.plan = self.bundle.root, self.bundle.plan
        self.manifest = json.loads((self.root / "analysis_manifest.json").read_text())
        if self.manifest["plan_fingerprint"] != self.plan["plan_fingerprint"] or self.manifest[
            "data_manifest_sha256"
        ] != sha(self.root / "data_manifest.json"):
            raise ProtocolError("analysis and observations manifests differ")
        report = json.loads((self.root / "report.json").read_text())
        if self.manifest.get("report_sha256") and self.manifest["report_sha256"] != sha(self.root / "report.json"):
            raise ProtocolError("analysis report changed")
        if report["missing"] or report["completed"] != report["planned"]:
            raise ValueError("publication plots require complete selected-task coverage")
        self.cache, self.used = {}, {}

    def rows(self, name, epsilon):
        key = (name, epsilon)
        if key not in self.cache:
            relative = "tables/" + name
            if relative not in self.manifest["tables"]:
                raise FileNotFoundError(
                    f"missing analysis table {name}; rerun analyze to compute the required artifact"
                )
            path = self.root / relative
            if relative not in self.used:
                digest = sha(path)
                if digest != self.manifest["tables"][relative]:
                    raise ProtocolError(f"analysis table changed: {relative}")
                self.used[relative] = digest
            self.cache[key] = read_table(path, epsilon)
        return self.cache[key]

    def has(self, name):
        return "tables/" + name in self.manifest["tables"]


def read_table(path, epsilon=None):
    rows = []
    for row in csv_rows(path):
        if epsilon is not None and "epsilon" in row and float(row["epsilon"]) != epsilon:
            continue
        parsed = {}
        for k, v in row.items():
            try:
                parsed[k] = float(v) if v != "" else np.nan
            except ValueError:
                parsed[k] = v
        rows.append(parsed)
    return rows


def condition_legend():
    return [
        Line2D([], [], color=COLORS[i], ls=ls, label=f"{LABELS[i]} / {'HK' if rule == 'hk' else 'Deffuant'}")
        for i in (0, 2)
        for rule, ls in zip(RULES, ("-", "--"))
    ]


def kernel_legend():
    return [Line2D([], [], color=c, label=label) for c, label in zip(COLORS, LABELS)]


def decorate(axes):
    for i, ax in enumerate(np.asarray(axes, dtype=object).flat):
        title = ax.get_title()
        ax.set_title("")
        ax.set_title(f"({chr(97 + i)}) {title}", loc="left", fontsize=8, pad=5)
        ax.spines[["top", "right"]].set_visible(False)
        if ax.axison:
            ax.grid(alpha=0.14, lw=0.5)
            ax.set_axisbelow(True)
        ax.yaxis.get_offset_text().set_x(0.18)
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(NullFormatter())


def light_strokes(fig, *, preserve_event=False):
    """Use lighter publication strokes without changing fonts, layouts or values."""
    preserved = fig.axes[0] if preserve_event else None

    def shrink(line):
        line.set_linewidth(line.get_linewidth() * 0.6)
        if line.get_marker() not in (None, "None", "none", "", " "):
            face = to_rgba(line.get_markerfacecolor())
            hollow = face[3] == 0 or all(v > 0.98 for v in face[:3])
            line.set_markersize(line.get_markersize() * (0.65 if hollow else 0.55))
            line.set_markeredgewidth(line.get_markeredgewidth() * 0.6)

    for ax in fig.findobj(match=Axes):
        if ax is preserved:
            continue
        for line in ax.lines:
            shrink(line)
        for collection in ax.collections:
            if isinstance(collection, LineCollection):
                collection.set_linewidth((np.asarray(collection.get_linewidth()) * 0.6).tolist())
            elif isinstance(collection, PathCollection):
                collection.set_sizes(collection.get_sizes() * 0.55**2)
                collection.set_linewidth((np.asarray(collection.get_linewidth()) * 0.6).tolist())
        for spine in ax.spines.values():
            spine.set_linewidth(spine.get_linewidth() * 0.75)
        for line in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            line.set_linewidth(line.get_linewidth() * 0.6)
        for tick in (
            *ax.xaxis.get_major_ticks(),
            *ax.xaxis.get_minor_ticks(),
            *ax.yaxis.get_major_ticks(),
            *ax.yaxis.get_minor_ticks(),
        ):
            for line in (tick.tick1line, tick.tick2line):
                line.set_markeredgewidth(line.get_markeredgewidth() * 0.75)
        legend = ax.get_legend()
        if legend:
            for line in legend.get_lines():
                shrink(line)
    for legend in fig.legends:
        for line in legend.get_lines():
            shrink(line)


def finish(fig, directory, name, caption, captions):
    light_strokes(fig, preserve_event=name == "figure_1_intuition")
    save(fig, directory / name)
    (directory / (name + ".txt")).write_text(caption + "\n")
    captions[str(directory / name)] = caption


def time_axis(ax, maximum=480):
    ax.set_xscale("symlog", linthresh=1)
    ticks = [v for v in (0, 1, 10, maximum) if v <= maximum]
    ticks = list(dict.fromkeys(ticks))
    ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
    ax.set(xlim=(0, maximum), xlabel=r"Scaled time $\tau$")


def event_network(ax):
    """Illustrative event, rendered by the same snapshot utility as the EHK paper."""
    opinions = np.array([-0.65, -0.55, -0.48, -0.42, -0.35, 0.35, 0.45, 0.55, 0.65])
    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (5, 6), (6, 7), (7, 8), (8, 5), (5, 7), (2, 6)])
    pos = nx.spring_layout(graph, seed=42, iterations=100)
    plot_network_snapshot(pos, opinions, graph, ax=ax, colorbar=False, node_size=25, alpha=0.26, rasterized=False)
    # A discordant followee is replaced by a concordant recommendation.
    nx.draw_networkx_edges(graph, pos, edgelist=[(2, 6)], ax=ax, edge_color="#bd4e19", style="dashed", width=1.4)
    nx.draw_networkx_edges(graph, pos, edgelist=[(2, 4)], ax=ax, edge_color="#166da8", style="dotted", width=1.8)
    for node, label in ((2, "i"), (6, "j"), (4, "k")):
        ax.annotate(label, pos[node], xytext=(3, 5), textcoords="offset points", fontsize=7)
    start = np.asarray(pos[2])
    end = start + 0.22 * (np.asarray(pos[4]) - start)
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.set_title("Event network")
    ax.set_axis_off()
    ax.set_box_aspect(1)


def intuition(data, epsilon, output, captions):
    nodes = data.rows("representative_nodes.csv", epsilon)
    edges = data.rows("representative_edges.csv.gz", epsilon)
    response = data.rows("representative_response.csv", epsilon)
    if not nodes or not edges or not response:
        raise ValueError("intuition needs representative fields and response; run analyze first")
    fig, axes = plt.subplots(1, 5, figsize=(7.1, 1.95), layout="constrained")
    event_network(axes[0])
    for source, color, ls in (("micro", ".2", "-"), ("pair", COLORS[2], "--")):
        rs = sorted([r for r in nodes if r["source"] == source], key=lambda r: r["bin"])
        axes[1].plot([r["x"] for r in rs], [r["density"] for r in rs], color=color, ls=ls, lw=1.1)
    axes[1].set(xlabel=r"Opinion $x$", ylabel=r"$\rho(x)$", title="Node snapshot", xticks=[-1, 0, 1])
    matrices = []
    for source in ("micro", "pair"):
        rs = [r for r in edges if r["source"] == source]
        bins = int(rs[0]["bins"])
        matrix = np.zeros((bins, bins))
        for r in rs:
            matrix[int(r["i"]), int(r["j"])] = r["density"]
        matrices.append(matrix)
    norm = PowerNorm(gamma=0.25, vmin=0, vmax=max(float(np.max(m)) for m in matrices))
    for ax, matrix, title in zip(axes[2:4], matrices, ("Microscopic $e$", "Measure-map $e$")):
        heat = ax.imshow(
            matrix.T,
            origin="lower",
            extent=(-1, 1, -1, 1),
            cmap="magma",
            norm=norm,
            interpolation="nearest",
            rasterized=True,
        )
        ax.set(xlabel=r"Source $x$", ylabel=r"Target $y$", xticks=[-1, 0, 1], yticks=[-1, 0, 1], title=title)
        ax.grid(False)
    # A thin inset scale keeps the two heatmaps square at full manuscript width.
    bar = axes[3].inset_axes([1.04, 0, 0.035, 1])
    fig.colorbar(heat, cax=bar, ticks=[0, 10, 100, 1000])
    bar.tick_params(labelsize=6, length=1.5, pad=1)
    for channel, color in (("node", COLORS[2]), ("edge", COLORS[4])):
        for source, ls in (("micro", "-"), ("pair", "--")):
            rs = sorted(
                [r for r in response if r["source"] == source and r["channel"] == channel], key=lambda r: r["tau"]
            )
            axes[4].plot([r["tau"] for r in rs], [r["remaining"] for r in rs], color=color, ls=ls, lw=1.05)
    time_axis(axes[4], 120)
    axes[4].set(ylim=(-0.05, 1.05), ylabel=r"$S_X$", title="Energy release")
    for ax in axes[1:]:
        ax.set_box_aspect(1)
    decorate(axes)
    for ax in axes[2:4]:
        ax.grid(False)
    handles = [
        Line2D([], [], color=".2", label="Microscopic", lw=1.2),
        Line2D([], [], color=".2", ls="--", label="Measure map", lw=1.2),
        Line2D([], [], color=COLORS[2], label=r"Node ($\rho$)", lw=1.2),
        Line2D([], [], color=COLORS[4], label=r"Edge ($E$)", lw=1.2),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    caption = (
        f"Figure 1. (a) Illustrative event network, drawn with the EHK snapshot tool: i follows discordant j "
        "(orange dashed edge); a concordant recommended k offers replacement (blue dotted edge), and the arrow "
        "illustrates opinion motion. This schematic is not a reconstructed graph or measured rate. "
        f"(b-d) HK Opinion4, epsilon={epsilon:g}, N=500, initial 0, tau=24; microscopic fields average four "
        "population-design processes; the map starts from the same empirical measures. Edge densities share "
        "a fourth-root color scale. (e) Four remaining-energy curves from that same ensemble and prediction, "
        "using each mean raw curve's own final-10% median on tau<=120; signed residuals are retained. "
        "Parameters: alpha=.02, q=.04, degree=15, recommended list=10, recommendation tolerance=.4, "
        "steepness=4, dt=1, D0=0, B=161, 33-point HK quadrature."
    )
    finish(fig, output, "figure_1_intuition", caption, captions)


def matched(data, epsilon, output, captions):
    protocol = data.plan["analysis_protocol"]
    rows = data.rows("matched_curves.csv", epsilon)
    areas = [
        r
        for r in data.rows("conditional_means.csv", epsilon)
        if matches(
            r,
            population=protocol["matched_population"],
            bins=protocol["matched_bins"],
            quadrature=protocol["matched_quadrature"],
            horizon=protocol["maximum_horizon_tau"],
            plateau_reference="own_endpoint",
        )
    ]
    fig = plt.figure(figsize=(7.1, 3.65), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
    top = [fig.add_subplot(gs[0, j]) for j in range(4)]
    bottom = [fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2:])]
    for ax, (rule, channel) in zip(top, ((r, c) for r in RULES for c in ("node", "edge"))):
        for idx in (0, 2, 4):
            rs = sorted(
                [
                    r
                    for r in rows
                    if matches(
                        r,
                        dynamics=rule,
                        channel=channel,
                        initial=0,
                        recommender=KERNELS[idx][0],
                        steepness=KERNELS[idx][1],
                    )
                ],
                key=lambda r: r["tau"],
            )
            t = [r["tau"] for r in rs]
            ax.fill_between(t, [r["lower"] for r in rs], [r["upper"] for r in rs], color=COLORS[idx], alpha=0.16, lw=0)
            ax.plot(t, [r["mean"] for r in rs], color=COLORS[idx], lw=1.2)
            ax.plot(t, [r["pair"] for r in rs], color=COLORS[idx], lw=1.05, ls="--")
        time_axis(ax)
        label = r"\rho" if channel == "node" else "E"
        ax.set(
            ylabel=rf"$\mathcal{{U}}_{label}/\mathcal{{E}}_\star$",
            title=("HK" if rule == "hk" else "Deffuant") + f" / {channel}",
        )
        ax.yaxis.set_major_locator(MaxNLocator(3))
    for ax, channel in zip(bottom, ("node", "edge")):
        for j, idx in enumerate((0, 2, 4)):
            for rule, offset in zip(RULES, (-0.16, 0.16)):
                for initial, marker in ((0, "o"), (1, "s")):
                    rs = [
                        r
                        for r in areas
                        if matches(
                            r, dynamics=rule, initial=initial, recommender=KERNELS[idx][0], steepness=KERNELS[idx][1]
                        )
                    ]
                    for r in rs:
                        x = j + offset + (initial - 0.5) * 0.1
                        ax.vlines(x, r[channel + "_lower"], r[channel + "_upper"], color=COLORS[idx], lw=1)
                        ax.plot(
                            x,
                            r[channel + "_mean"],
                            marker,
                            mec=COLORS[idx],
                            mfc=COLORS[idx] if rule == "hk" else "white",
                            ms=3.5,
                        )
                        ax.plot(x + 0.035, r["pair_" + channel + "_mean"], "x", color=COLORS[idx], ms=4)
        symbol = r"\rho" if channel == "node" else "E"
        ax.set(
            xticks=range(3),
            xticklabels=[LABELS[i] for i in (0, 2, 4)],
            ylabel=rf"$\widehat\Theta_{symbol}$",
            title=f"{channel.capitalize()} residual area",
            xlim=(-0.45, 2.45),
            yscale="log",
        )
    decorate([*top, *bottom])
    handles = [Line2D([], [], color=COLORS[i], label=LABELS[i]) for i in (0, 2, 4)]
    handles += [
        Line2D([], [], color=".3", ls=ls, label=label) for ls, label in (("-", "Microscopic"), ("--", "Measure map"))
    ]
    handles += [
        Line2D([], [], color=".3", ls="", marker=m, label=label)
        for m, label in (("o", "Initial 0"), ("s", "Initial 1"), ("x", "Map area"))
    ]
    handles += [
        Line2D([], [], color=".3", ls="", marker="o", mfc=fill, label=label)
        for fill, label in ((".3", "HK (filled)"), ("white", "Deffuant (hollow)"))
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    caption = (
        f"Figure 2. Matched microscopic and measure-map dynamics, epsilon={epsilon:g}, N=500, r=2. "
        "Top: raw node/edge energies for HK and Deffuant, initial 0, with eight conditional processes and 95% "
        "whole-process bootstrap bands. Bottom: node and edge areas for both initial graphs; circles/squares "
        "denote initial 0/1, filled/hollow HK/Deffuant, crosses predictions. The bootstrap conditions on each "
        "initial state; processes are not independent initial states. Own-reference areas use each mean curve's "
        f"final-10% median through tau={protocol['maximum_horizon_tau']:g}, recomputed in 1000 resamples. "
        "B=161 and 33-point HK quadrature. These are finite-window areas."
    )
    finish(fig, output, "figure_2_matched", caption, captions)


def scaling(data, output, captions):
    epsilons = data.plan["analysis_protocol"].get("confidence_radii", [0.4])
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.75), layout="constrained")
    alphas = sorted({t["parameters"]["alpha"] for t in data.plan["tasks"] if "scaling" in t["roles"]})
    eps_colors = ["#0072b2", "#d55e00", "#009e73"]
    for row, rule in enumerate(RULES):
        for col, channel in ((0, "node"), (2, "edge")):
            ax = axes[row, col]
            for epsilon, color in zip(epsilons, eps_colors):
                tasks = [
                    t
                    for t in data.plan["tasks"]
                    if t["kind"] == "measure"
                    and "scaling" in t["roles"]
                    and matches(
                        t["parameters"],
                        dynamics=rule,
                        epsilon=epsilon,
                        recommender="opinion_random",
                        steepness=4.0,
                        ratio=2.0,
                    )
                ]
                cases = sorted([data.bundle.case(t, fields=False) for t in tasks], key=lambda c: c["alpha"])
                if not cases:
                    continue
                reference, largest = cases[0], cases[-1]
                name = channel + "_energy"
                # The envelope is a display range, not a statistical uncertainty band.
                # Interpolate only inside every curve's recorded interval.
                stop = min(c["tau"][-1] for c in cases)
                common = reference["tau"][reference["tau"] <= stop]
                values = np.array([np.interp(common, c["tau"], c[name]) for c in cases])
                ax.fill_between(common, values.min(0), values.max(0), color=color, alpha=0.22, lw=0)
                ax.plot(reference["tau"], reference[name], color=color, lw=1.3)
                targets = np.geomspace(max(largest["tau"][1], 0.1), largest["tau"][-1], 9)
                indices = np.unique([np.argmin(np.abs(largest["tau"] - at)) for at in targets])
                ax.plot(
                    largest["tau"][indices],
                    largest[name][indices],
                    ls="",
                    marker="o",
                    mfc="white",
                    mec=color,
                    ms=2.8,
                    mew=0.8,
                )
            time_axis(ax)
            symbol = r"\rho" if channel == "node" else "E"
            ax.set(ylabel=rf"$\mathcal{{U}}_{symbol}/\mathcal{{E}}_\star$")
            ax.set_title(("HK" if rule == "hk" else "Deffuant") + f" / {channel}")
            error = axes[row, col + 1]
            for epsilon, color in zip(epsilons, eps_colors):
                rs = [
                    r
                    for r in data.rows("scaling.csv", epsilon)
                    if matches(r, dynamics=rule, energy=channel + "_energy")
                ]
                updates = sorted({r["alpha"] for r in rs})
                error.plot(
                    updates,
                    [100 * max(r["relative_max"] for r in rs if r["alpha"] == a) for a in updates],
                    "o-",
                    ms=2.5,
                    color=color,
                    lw=1.1,
                )
            error.set(
                xscale="log",
                yscale="log",
                xlabel=r"Update size $\alpha$",
                ylabel=f"{channel.capitalize()} gap (%)",
                title="Maximum error",
            )
            updates = sorted(set(alphas) - {min(alphas)})
            error.set_xticks([updates[0], updates[-1]], [f"{updates[0]:g}", f"{updates[-1]:g}"])
    decorate(axes)
    handles: list[Line2D | Patch] = [
        Line2D([], [], color=c, label=rf"$\epsilon={e:g}$") for e, c in zip(epsilons, eps_colors)
    ]
    handles += [
        Line2D([], [], color=".3", lw=1.3, label=rf"$\alpha={min(alphas):g}$"),
        Line2D([], [], color=".3", ls="", marker="o", mfc="white", ms=3, label=rf"$\alpha={max(alphas):g}$"),
        Patch(facecolor=".5", alpha=0.22, label="Range of all updates"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    caption = (
        "Figure 3. Rows: HK, Deffuant; columns: raw node energy, node error, raw edge energy, edge error. "
        "Colors identify epsilon=.2,.4,.8. Solid curves use alpha=.0025, hollow markers alpha=.04; "
        "shading spans all five updates on the common observed interval, with linear display interpolation "
        "and no extrapolation. It is a calculated range, not a confidence band. "
        "Representative paths use Opinion4, r=2, with q=r*alpha, uniform initial node measure and random mixing. "
        "Errors maximize the raw-energy gap over all five recommendation laws and r=.25,2,10, separately "
        "for each epsilon, rule and energy. Each comparison uses alpha_ref=.0025, its common observed interval, "
        "and the larger of the reference initial magnitude and range as denominator. No alignment or tail "
        "extension. B=161, 33-point HK quadrature; spatial error is assessed separately."
    )
    finish(fig, output, "figure_3_scaling", caption, captions)


def rate_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.05, 45)
    ax.set_xticks([0.0625, 0.25, 1, 4, 16, 32], ["0", ".25", "1", "4", "16", "32"])
    ax.axvline(0.0884, color=".7", ls=":", lw=0.7)
    ax.set_xlabel(r"Relative rate $r=q/\alpha$")


def response_curve(ax, rs, value, flag, color, ls="-"):
    rs = sorted(rs, key=lambda r: r["ratio"])
    xs = np.array([r["ratio"] if r["ratio"] else 0.0625 for r in rs])
    ys = np.array([r[value] for r in rs])
    good = np.array([r[flag] == "True" for r in rs])
    positive = np.array([r["ratio"] > 0 for r in rs])
    ax.plot(xs[positive], np.where(good[positive], ys[positive], np.nan), color=color, ls=ls, lw=1.15)
    for valid, fill in ((True, color), (False, "white")):
        pick = (good == valid) & np.isfinite(ys)
        ax.plot(xs[pick], ys[pick], "o", mfc=fill, mec=color, ms=3)


def recommendation(data, output, captions, *, relative=False):
    epsilons = data.plan["analysis_protocol"].get("confidence_radii", [0.4])
    fig, axes = plt.subplots(2, len(epsilons), figsize=(7.1, 3.7), layout="constrained", squeeze=False)
    styles = ["-", ":", "--", "-.", (0, (5, 1, 1, 1))]
    for col, epsilon in enumerate(epsilons):
        rows = data.rows("recommendation.csv", epsilon)
        for row, rule in enumerate(RULES):
            ax = axes[row, col]
            for i, (rec, z) in enumerate(KERNELS):
                rs = [r for r in rows if matches(r, dynamics=rule, recommender=rec, steepness=z)]
                response_curve(
                    ax,
                    rs,
                    "shift" if relative else "delta",
                    "shift_confirmed" if relative else "confirmed",
                    COLORS[i],
                    styles[i],
                )
            rate_axis(ax)
            if relative:
                baseline = sorted(
                    [r for r in rows if matches(r, dynamics=rule, recommender="random", steepness=1.0)],
                    key=lambda r: r["ratio"],
                )
                xs = [r["ratio"] if r["ratio"] else 0.0625 for r in baseline]
                equal_area = np.array([-r["delta"] for r in baseline])
                good = np.array([r["confirmed"] == "True" for r in baseline])
                positive = np.array([r["ratio"] > 0 for r in baseline])
                ax.plot(
                    np.array(xs)[positive],
                    np.where(good[positive], equal_area[positive], np.nan),
                    color=".25",
                    ls=(0, (1, 2)),
                    lw=1.0,
                )
                ax.axhline(0, color=COLORS[0], lw=0.7)
            else:
                ax.axhline(0, color=".4", lw=0.7)
            ax.set(
                ylabel=r"$\Delta_a-\Delta_{\rm Random}$"
                if relative
                else r"$\log(\widehat\Theta_E/\widehat\Theta_\rho)$",
                title=("HK" if rule == "hk" else "Deffuant") + rf", $\epsilon={epsilon:g}$",
            )
    decorate(axes)
    handles = [Line2D([], [], color=c, ls=ls, label=label) for c, ls, label in zip(COLORS, styles, LABELS)]
    if relative:
        handles.append(Line2D([], [], color=".25", ls=(0, (1, 2)), label=r"Equal areas ($\Delta_a=0$)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3 if relative else 5, frameon=False)
    caption = (
        "Figure 4. Rows: HK, Deffuant; columns: epsilon=.2,.4,.8. All panels show the log ratio of observed "
        "edge and node residual-energy areas under the five recommendation laws. Own-final-10%-median "
        "references and each trajectory's actual horizon (tau=159.8-480 in the formal design) are retained. "
        "Hollow markers flag either unconfirmed plateau and connecting segments break there; q=0 is isolated "
        "at the left. Alpha=.02, degree=15, recommendations=10, tolerance=.4, dt=1, D0=0, B=161, 33-point HK "
        "quadrature, uniform initial node measure and random mixing. Sign changes concern finite-window energy "
        "areas; they do not determine infinite-time relaxation or polarization/segregation formation order."
    )
    if relative:
        caption = caption.replace(
            "All panels show the log ratio of observed edge and node residual-energy areas under the five recommendation laws.",
            "All panels show each recommendation's log-area-ratio difference from Random at the same rate and configuration.",
        )
        caption += (
            " Alternative coordinates: subtract the Random log ratio at each identical rate and "
            "configuration. Random is therefore flat at zero; the original equal-area line moves "
            "to minus the Random log ratio. Confirmation flags also require the Random reference."
        )
    finish(
        fig,
        output,
        "figure_4_recommendation_random_baseline" if relative else "figure_4_recommendation",
        caption,
        captions,
    )


def population(data, epsilon, output, captions):
    structure = data.rows("population_structure.csv", epsilon)
    variation = data.rows("population_variation.csv", epsilon)
    at = data.plan["analysis_protocol"]["population_horizon_tau"]
    panels = [
        (structure, "opinion_variance", "mean", "lower", "upper", "Continuous opinion\nvariance"),
        (structure, "excess_concordance", "mean", "lower", "upper", "Excess concordance"),
        (
            variation,
            "node_cdf",
            "process_variance",
            "process_variance_loio_min",
            "process_variance_loio_max",
            "Node CDF\nprocess variance W",
        ),
        (
            variation,
            "node_cdf",
            "initial_variance_corrected",
            "initial_variance_corrected_loio_min",
            "initial_variance_corrected_loio_max",
            "Node CDF\ncorrected initial variation I",
        ),
        (
            variation,
            "edge_cdf",
            "process_variance",
            "process_variance_loio_min",
            "process_variance_loio_max",
            "Joint edge CDF\nprocess variance W",
        ),
        (
            variation,
            "edge_cdf",
            "initial_variance_corrected",
            "initial_variance_corrected_loio_min",
            "initial_variance_corrected_loio_max",
            "Joint edge CDF\ncorrected initial variation I",
        ),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 5.35), layout="constrained")
    for ax, (source, metric, point, low, high, label) in zip(axes.flat, panels):
        for idx in (0, 2):
            for rule, ls in zip(RULES, ("-", "--")):
                rs = sorted(
                    [
                        r
                        for r in source
                        if matches(
                            r,
                            metric=metric,
                            recommender=KERNELS[idx][0],
                            steepness=KERNELS[idx][1],
                            dynamics=rule,
                            tau=at,
                        )
                    ],
                    key=lambda r: r["population"],
                )
                ns = [r["population"] for r in rs]
                ax.plot(ns, [r[point] for r in rs], "o" + ls, color=COLORS[idx], ms=4)
                ax.fill_between(ns, [r[low] for r in rs], [r[high] for r in rs], color=COLORS[idx], alpha=0.10)
        ns = sorted({r["population"] for r in source})
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [f"{n:g}" for n in ns])
        ax.set(xlabel="Population N", ylabel=label)
        ax.axhline(0, color=".65", lw=0.65)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axes[0, 0].set_title("Initial-group bootstrap")
    axes[0, 1].set_title("Initial-group bootstrap")
    for ax in axes[1:].flat:
        ax.set_title("LOIO sensitivity range")
    decorate(axes)
    fig.legend(handles=condition_legend(), loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    caption = (
        f"Figure 5. Population design at fixed r=2 and tau={at:g}, Random/Opinion4 and both update rules. "
        "Panels a,b show means of microscopic observables, with exploratory bootstrap intervals resampling entire "
        "initial-state groups. Panels c–f use weighted node/joint-edge CDF norms; shading is the leave-one-initial-out "
        "minimum–maximum range, not a 95% confidence interval. All branches, including near-consensus outcomes, "
        "are retained. Corrected I may be negative. Near-consensus locations can still differ in CDF coordinates; "
        "floating-point-level corrections are not evidence of physical amplification. "
        "The tested sizes do not establish an asymptotic population law."
    )
    finish(fig, output, "figure_5_population", caption, captions)


def supplementary(data, epsilon, output, captions, selected):
    if "recommendation" in selected:
        rows = data.rows("mean_release.csv", epsilon)
        rows = [r for r in rows if "effects" in r["roles"].split(",") and r["is_observed_endpoint"] == "True"]
        fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.0), layout="constrained")
        for col, rule in enumerate(RULES):
            for row, channel in enumerate(("node", "edge")):
                ax = axes[row, col]
                for idx, (rec, z) in enumerate(KERNELS):
                    rs = [r for r in rows if matches(r, dynamics=rule, channel=channel, recommender=rec, steepness=z)]
                    response_curve(ax, rs, "mean_tau", "plateau_confirmed", COLORS[idx])
                rate_axis(ax)
                ax.set_ylabel(f"{channel.capitalize()} residual area")
                ax.set_yscale("log")
                if row == 0:
                    ax.set_title(rule.upper())
        decorate(axes)
        fig.legend(handles=kernel_legend(), loc="outside lower center", ncol=5, frameon=False)
        finish(
            fig,
            output,
            "absolute_areas",
            "Supplement: absolute finite-window own-endpoint residual areas; hollow symbols mark unconfirmed plateaus.",
            captions,
        )
    if "numerics" in selected:
        numerical_controls(data, epsilon, output, captions)
    if "population" not in selected:
        return
    rows = data.rows("population_variation.csv", epsilon)
    for n in sorted({r["population"] for r in rows}):
        fig, axes = plt.subplots(3, 2, figsize=(7.1, 5.5), layout="constrained")
        for col, metric in enumerate(("node_cdf", "edge_cdf")):
            for ax, component, label in zip(
                axes[:, col],
                ("process_variance", "initial_variance_corrected", "conditional_bias_squared_corrected"),
                ("W", "I", "B²"),
            ):
                for idx in (0, 2):
                    for rule, ls in zip(RULES, ("-", "--")):
                        rs = sorted(
                            [
                                r
                                for r in rows
                                if matches(
                                    r,
                                    metric=metric,
                                    population=n,
                                    dynamics=rule,
                                    recommender=KERNELS[idx][0],
                                    steepness=KERNELS[idx][1],
                                )
                            ],
                            key=lambda r: r["tau"],
                        )
                        t = [r["tau"] for r in rs]
                        ax.plot(t, [r[component] for r in rs], "o" + ls, color=COLORS[idx], ms=3)
                        ax.fill_between(
                            t,
                            [r[component + "_loio_min"] for r in rs],
                            [r[component + "_loio_max"] for r in rs],
                            color=COLORS[idx],
                            alpha=0.1,
                        )
                ax.set(xlabel=r"$\tau$", ylabel=metric.replace("_", " ") + " " + label)
                ax.axhline(0, color=".6", lw=0.65)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        decorate(axes)
        fig.legend(handles=condition_legend(), loc="outside lower center", ncol=4, frameon=False, fontsize=7)
        fig.suptitle(rf"Population N={n:g}: variation and prediction residuals; $\epsilon={epsilon:g}$")
        finish(
            fig,
            output,
            f"variation_time_N{n:g}",
            "Supplement: W, corrected I and finite-N conditional residual B² at all recorded field snapshots. Bands are LOIO ranges, not confidence intervals. MSE=W+B²; I is not another prediction-error term. Negative corrections are retained.",
            captions,
        )


def terminal_variation(data, output, captions):
    """Manuscript-size CDF panels; no small-font pages imported from an atlas."""
    epsilons = data.plan["analysis_protocol"].get("confidence_radii", [0.4])
    at = data.plan["analysis_protocol"]["population_horizon_tau"]
    components = (
        ("process_variance", r"$W$"),
        ("initial_variance_corrected", r"$I$"),
        ("conditional_bias_squared_corrected", r"$B^2$"),
    )
    for metric, name in (("node_cdf", "node_cdf_terminal"), ("edge_cdf", "edge_cdf_terminal")):
        fig, axes = plt.subplots(3, len(epsilons), figsize=(7.1, 5.6), layout="constrained", squeeze=False)
        for col, epsilon in enumerate(epsilons):
            rows = [r for r in data.rows("population_variation.csv", epsilon) if matches(r, metric=metric, tau=at)]
            for row, (component, label) in enumerate(components):
                ax = axes[row, col]
                for idx in (0, 2):
                    for rule, ls in zip(RULES, ("-", "--")):
                        rs = sorted(
                            [
                                r
                                for r in rows
                                if matches(r, dynamics=rule, recommender=KERNELS[idx][0], steepness=KERNELS[idx][1])
                            ],
                            key=lambda r: r["population"],
                        )
                        ns = [r["population"] for r in rs]
                        ax.plot(ns, [r[component] for r in rs], "o" + ls, color=COLORS[idx], ms=2.5, lw=1.1)
                        ax.fill_between(
                            ns,
                            [r[component + "_loio_min"] for r in rs],
                            [r[component + "_loio_max"] for r in rs],
                            color=COLORS[idx],
                            alpha=0.1,
                        )
                ax.set_xscale("log", base=2)
                ns = sorted({r["population"] for r in rows})
                ax.set_xticks(ns, [f"{n:g}" for n in ns])
                ax.set_yscale("symlog", linthresh=1e-4)
                ax.axhline(0, color=".6", lw=0.6)
                ax.set_ylabel(label if col == 0 else "")
                if row == 0:
                    ax.set_title(rf"$\epsilon={epsilon:g}$")
                if row == 2:
                    ax.set_xlabel(r"Population $N$")
        decorate(axes)
        fig.legend(handles=condition_legend(), loc="outside lower center", ncol=4, frameon=False)
        caption = (
            f"Supplement. {metric.replace('_', ' ')} at tau={at:g}: rows W, corrected I and corrected "
            "finite-N conditional residual B²; columns are confidence radii. Shading is the LOIO range, "
            "not a confidence interval. Signed log axes retain negative corrections with a linear region "
            "at absolute values below 1e-4. Norm weights and the population protocol are unchanged."
        )
        finish(fig, output, name, caption, captions)


def numerical_controls(data, epsilon, output, captions):
    rows = data.rows("numerics.csv", epsilon)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.0), layout="constrained")
    for col, rule in enumerate(RULES):
        rs = [r for r in rows if r["dynamics"] == rule]
        controls = sorted({(r["bins"], r["quadrature"], r["dt"]) for r in rs})
        for row, (metric, label) in enumerate(
            (("relative_max", "Raw-energy maximum gap (%)"), ("mean_relative_change", "Residual-area change (%)"))
        ):
            matrix = np.full((len(controls), 4), np.nan)
            for i, (bins, q, dt) in enumerate(controls):
                for j, (rec, z, channel) in enumerate(
                    (
                        ("opinion_random", 4.0, "node"),
                        ("opinion_random", 4.0, "edge"),
                        ("structure_random_l0", 4.0, "node"),
                        ("structure_random_l0", 4.0, "edge"),
                    )
                ):
                    hits = [
                        r
                        for r in rs
                        if matches(r, bins=bins, quadrature=q, dt=dt, recommender=rec, steepness=z, channel=channel)
                    ]
                    if hits:
                        matrix[i, j] = 100 * hits[0][metric]
            ax = axes[row, col]
            if controls:
                heat = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="YlOrRd")
                fig.colorbar(heat, ax=ax, label=label, shrink=0.8)
                ax.set(
                    xticks=range(4),
                    xticklabels=["Opinion4\nnode", "Opinion4\nedge", "Structure4\nnode", "Structure4\nedge"],
                    yticks=range(len(controls)),
                    yticklabels=[f"B={b:g}, Q={q:g}, dt={dt:g}" for b, q, dt in controls],
                )
                ax.tick_params(axis="x", labelsize=8)
            ax.set_title(rule.upper())
    decorate(axes)
    fig.suptitle(rf"Independent numerical controls; $\epsilon={epsilon:g}$")
    finish(
        fig,
        output,
        "numerical_controls",
        "Supplement: single-factor spatial, quadrature and physical-step controls. Endpoints and final fields belong to their own observed windows; these controls are separate from fixed-r small-update similarity.",
        captions,
    )


def plot(directory, output=None):
    data = Analysis(directory)
    output = Path(output).resolve() if output else data.root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    setup_paper_params()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "figure.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    selected = topics(data.plan["profile"])
    baseline = data.plan["analysis_protocol"]["baseline_epsilon"]
    captions: dict[str, str] = {}
    for epsilon in data.plan["analysis_protocol"].get("confidence_radii", [baseline]):
        destination = output if epsilon == baseline else output / "supplementary" / f"epsilon_{epsilon:g}"
        destination.mkdir(parents=True, exist_ok=True)
        if "population" in selected:
            intuition(data, epsilon, destination, captions)
        if "matched" in selected:
            matched(data, epsilon, destination, captions)
        if "population" in selected:
            population(data, epsilon, destination, captions)
        supplement = destination if epsilon != baseline else output / "supplementary"
        supplement.mkdir(exist_ok=True)
        supplementary(data, epsilon, supplement, captions, selected)
        data.cache.clear()
    if "population" in selected:
        terminal_variation(data, output / "supplementary", captions)
    if "scaling" in selected:
        scaling(data, output, captions)
    if "recommendation" in selected:
        recommendation(data, output, captions)
        recommendation(data, output, captions, relative=True)
    manifest = {
        "schema": 1,
        "plan_fingerprint": data.plan["plan_fingerprint"],
        "profile": data.plan["profile"],
        "display_style": {
            "data_strokes": 0.6,
            "filled_marker_radius": 0.55,
            "hollow_marker_radius": 0.65,
            "frame_tick_strokes": 0.75,
            "figure_1_event_panel": "unchanged",
        },
        "research_topics": list(selected),
        "input_scope": "portable local CSV only; no raw path lookup, solver, bootstrap or fitting",
        "analysis_manifest_sha256": sha(data.root / "analysis_manifest.json"),
        "inputs": {**data.used, **data.bundle.used},
        "renderer_sources": {
            str(path.relative_to(Path(__file__).resolve().parents[3])): sha(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("figure_style.py"),
                Path(inspect.getfile(setup_paper_params)),
                Path(inspect.getfile(save)),
                Path(inspect.getfile(plot_network_snapshot)),
            )
        },
        "figures": [str(Path(p).relative_to(output)) + ".pdf" for p in sorted(captions)],
        "output_sha256": {
            str(Path(p).relative_to(output)) + ext: sha(Path(p + ext))
            for p in sorted(captions)
            for ext in (".pdf", ".png", ".txt")
        },
        "captions": {str(Path(k).relative_to(output)): v for k, v in captions.items()},
        "smoke_is_scientific_evidence": False,
        "smoke": is_smoke(data.plan["profile"]),
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(data.root / "figure_manifest.json", manifest)
    return output
