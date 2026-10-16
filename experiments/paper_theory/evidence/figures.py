"""One rendering implementation for the revision figures, exclusively from local CSV."""

import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

from common.io.json import atomic_json, sha
from common.plotting import setup_paper_params
from common.plotting.figures import decorate, save
from common.runtime.failures import ProtocolError

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
        Line2D([], [], color=COLORS[i], ls=ls, label=f"{LABELS[i]} / {rule.upper()}")
        for i in (0, 2)
        for rule, ls in zip(RULES, ("-", "--"))
    ]


def kernel_legend():
    return [Line2D([], [], color=c, label=label) for c, label in zip(COLORS, LABELS)]


def finish(fig, directory, name, caption, captions):
    for ax in fig.axes:
        # Keep scientific-notation offsets clear of the common panel labels.
        ax.yaxis.get_offset_text().set_x(0.12)
    save(fig, directory / name)
    (directory / (name + ".txt")).write_text(caption + "\n")
    captions[str(directory / name)] = caption


def intuition(data, epsilon, output, captions):
    nodes = [r for r in data.rows("representative_nodes.csv", epsilon) if r["source"] == "micro"]
    edges = [r for r in data.rows("representative_edges.csv.gz", epsilon) if r["source"] == "micro"]
    if not nodes or not edges:
        raise ValueError("intuition figure requires the declared population snapshot")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), layout="constrained")
    for ax in axes[0]:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    ax = axes[0, 0]
    for x, y, label, color in (
        (0.18, 0.52, r"$i$", COLORS[2]),
        (0.76, 0.76, r"$j$", COLORS[0]),
        (0.76, 0.25, r"$k$", COLORS[4]),
    ):
        ax.add_patch(Circle((x, y), 0.065, facecolor=color, edgecolor="white"))
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=14)
    for end in ((0.69, 0.73), (0.69, 0.28)):
        ax.add_patch(FancyArrowPatch(end, (0.26, 0.52), arrowstyle="->", mutation_scale=15, lw=1.5, color=".35"))
    ax.text(0.47, 0.85, "Followee exposure", ha="center", fontsize=11)
    ax.text(0.50, 0.15, "Recommended exposure", ha="center", fontsize=11)
    ax.annotate("", xy=(0.37, 0.48), xytext=(0.22, 0.48), arrowprops={"arrowstyle": "->", "color": COLORS[2], "lw": 3})
    ax.text(0.20, 0.08, r"Opinion influence: $x_i \mapsto x_i+\Delta x_i$", fontsize=12)
    ax.set_title("Opinion influence")
    ax = axes[0, 1]
    for x, y, label, color in (
        (0.18, 0.52, r"$i$", COLORS[2]),
        (0.77, 0.78, r"$j$", COLORS[0]),
        (0.77, 0.25, r"$k$", COLORS[4]),
    ):
        ax.add_patch(Circle((x, y), 0.065, facecolor=color, edgecolor="white"))
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=14)
    ax.annotate(
        "", xy=(0.70, 0.75), xytext=(0.25, 0.54), arrowprops={"arrowstyle": "->", "ls": "--", "color": ".6", "lw": 1.5}
    )
    ax.annotate(
        "", xy=(0.70, 0.28), xytext=(0.25, 0.48), arrowprops={"arrowstyle": "->", "color": COLORS[4], "lw": 2.5}
    )
    ax.text(0.48, 0.80, "Replace", ha="center", color=".4")
    ax.text(0.52, 0.12, "Acceptable recommended target", ha="center", fontsize=11)
    ax.set_title("Connection replacement")
    ax = axes[1, 0]
    nodes.sort(key=lambda r: r["bin"])
    ax.plot([r["x"] for r in nodes], [r["density"] for r in nodes], color=COLORS[2], lw=2)
    ax.set(xlabel="Opinion x", ylabel=r"Node density $\rho(x)$", title="Opinion differentiation")
    bins = int(nodes[0]["bins"])
    matrix = np.zeros((bins, bins))
    for r in edges:
        matrix[int(r["i"]), int(r["j"])] = r["density"]
    positive = matrix[matrix > 0]
    lower, upper = float(np.min(positive)), float(np.max(positive))
    heat = axes[1, 1].imshow(
        np.ma.masked_less_equal(matrix.T, 0),
        origin="lower",
        extent=(-1, 1, -1, 1),
        cmap=plt.get_cmap("magma").with_extremes(bad="#110b20"),
        norm=LogNorm(vmin=min(lower, upper / 10), vmax=upper),
        interpolation="nearest",
        aspect="equal",
    )
    axes[1, 1].set(xlabel="Source opinion x", ylabel="Target opinion y", title="Opinion-conditioned edges")
    fig.colorbar(heat, ax=axes[1, 1], label="Normalized edge density", shrink=0.8)
    decorate(axes)
    fig.suptitle(rf"Coupled node and edge dynamics; $\epsilon={epsilon:g}$")
    r = nodes[0]
    caption = (
        f"Figure 1. Upper panels are event schematics, not measured rates. Lower panels: HK Opinion4, "
        f"N={r['population']:g}, initial {r['initial']:g}, tau={r['tau']:g}; microscopic distributions average "
        f"{r['processes']:g} conditional processes from the population design. Directed-edge mass is normalized to one; "
        "the edge color scale is logarithmic and empty bins use the background color. "
        "These data are distinct from the long-window matched ensemble; this example is not a stability claim."
    )
    finish(fig, output, "figure_1_intuition", caption, captions)


def matched(data, epsilon, output, captions):
    protocol = data.plan["analysis_protocol"]
    rows = data.rows("matched_curves.csv", epsilon)
    areas = [
        r
        for r in data.rows("conditional_means.csv", epsilon)
        if r["population"] == protocol["matched_population"]
        and r["bins"] == protocol["matched_bins"]
        and r["quadrature"] == protocol["matched_quadrature"]
        and r["horizon"] == protocol["maximum_horizon_tau"]
        and r["plateau_reference"] == "own_endpoint"
    ]
    kernels = [
        i for i in (0, 2, 4) if any(matches(r, recommender=KERNELS[i][0], steepness=KERNELS[i][1]) for r in rows)
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11.8, 9.5), layout="constrained")
    for col, rule in enumerate(RULES):
        for channel, row in (("node", 0), ("edge", 1)):
            ax = axes[row, col]
            for idx in kernels:
                for initial in sorted({r["initial"] for r in rows}):
                    rs = sorted(
                        [
                            r
                            for r in rows
                            if matches(
                                r,
                                dynamics=rule,
                                recommender=KERNELS[idx][0],
                                steepness=KERNELS[idx][1],
                                initial=initial,
                                channel=channel,
                            )
                        ],
                        key=lambda r: r["tau"],
                    )
                    if not rs:
                        continue
                    tau = [r["tau"] for r in rs]
                    ax.plot(tau, [r["mean"] for r in rs], color=COLORS[idx], ls="-" if initial == 0 else ":", lw=1.5)
                    ax.fill_between(
                        tau, [r["lower"] for r in rs], [r["upper"] for r in rs], color=COLORS[idx], alpha=0.12
                    )
                    ax.plot(tau, [r["pair"] for r in rs], color=COLORS[idx], ls="--" if initial == 0 else "-.", lw=1)
            ax.set(xlabel=r"$\tau=\alpha t$", ylabel=f"Raw {channel} energy")
            if row == 0:
                ax.set_title(rule.upper())
    for ax, channel in zip(axes[2], ("node", "edge")):
        for j, idx in enumerate(kernels):
            for r in areas:
                if (r["recommender"], r["steepness"]) != KERNELS[idx]:
                    continue
                rule_index = RULES.index(r["dynamics"])
                initial = r["initial"]
                x = j + (-0.12 if rule_index == 0 else 0.12) + (initial - 0.5) * 0.07
                ax.vlines(x, r[channel + "_lower"], r[channel + "_upper"], color=COLORS[idx], lw=1)
                ax.plot(
                    x,
                    r[channel + "_mean"],
                    "o" if initial == 0 else "s",
                    mec=COLORS[idx],
                    mfc=COLORS[idx] if rule_index == 0 else "white",
                    ms=4,
                )
                ax.plot(x, r["pair_" + channel + "_mean"], "x", color=COLORS[idx], ms=4)
        ax.set(
            xticks=range(len(kernels)),
            xticklabels=[LABELS[i] for i in kernels],
            ylabel=f"{channel.capitalize()} residual area",
            title="Finite-window residual area",
        )
    decorate(axes)
    handles = [Line2D([], [], color=COLORS[i], label=LABELS[i]) for i in kernels]
    handles += [
        Line2D([], [], color="k", ls=ls, label=label)
        for ls, label in (
            ("-", "Micro initial 0"),
            (":", "Micro initial 1"),
            ("--", "Pair initial 0"),
            ("-.", "Pair initial 1"),
        )
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(rf"Matched microscopic and measure-map dynamics; $\epsilon={epsilon:g}$")
    caption = (
        f"Figure 2. Raw node/edge energies and own-endpoint finite-window residual areas through tau="
        f"{protocol['maximum_horizon_tau']:g}. Curves and bands use conditional process bootstrap within each fixed initial "
        f"({protocol['confidence_level']:.0%}); they do not treat process replicates as independent initial states. "
        "Area markers: circles/squares are initial 0/1, filled/hollow are HK/Deffuant, crosses are matched predictions. "
        "Agreement and systematic discrepancies are both retained; these areas are not infinite-time relaxation constants."
    )
    finish(fig, output, "figure_2_matched", caption, captions)


def scaling(data, epsilon, output, captions):
    selected = [
        t
        for t in data.plan["tasks"]
        if t["kind"] == "measure"
        and "scaling" in t["roles"]
        and matches(t["parameters"], epsilon=epsilon, recommender="opinion_random", steepness=4.0, ratio=2.0)
    ]
    fig, axes = plt.subplots(
        3, 2, figsize=(11.8, 10.4), gridspec_kw={"height_ratios": [1, 1, 1.8]}, layout="constrained"
    )
    alphas = sorted({t["parameters"]["alpha"] for t in selected})
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.85, max(1, len(alphas))))
    selected.sort(key=lambda t: t["parameters"]["alpha"], reverse=True)
    for t in selected:
        case = data.bundle.case(t, fields=False)
        col = RULES.index(t["parameters"]["dynamics"])
        for row, channel in enumerate(("node", "edge")):
            axes[row, col].plot(
                case["tau"],
                case[channel + "_energy"],
                color=colors[alphas.index(case["alpha"])],
                label=rf"$\alpha={case['alpha']:g}$",
            )
            axes[row, col].set(xlabel=r"$\tau=\alpha t$", ylabel=f"Raw {channel} energy")
            if row == 0:
                axes[row, col].set_title(RULES[col].upper() + ", Opinion4, r=2")
    if selected:
        axes[0, 0].legend(frameon=False, fontsize=9)
    rows = data.rows("scaling_display.csv", epsilon) if data.has("scaling_display.csv") else []
    values = [100 * r["relative_max"] for r in rows if r["relative_max"] > 0]
    norm = LogNorm(vmin=max(min(values, default=1e-4), 1e-9), vmax=max(max(values, default=1), 1e-3))
    heat = None
    for col, rule in enumerate(RULES):
        rs = [r for r in rows if r["dynamics"] == rule]
        configs = [
            (rec, z, r)
            for rec, z in KERNELS
            for r in sorted({v["r"] for v in rs})
            if any(matches(v, recommender=rec, steepness=z, r=r) for v in rs)
        ]
        xs = sorted({r["alpha"] for r in rs})
        if not configs or not xs:
            axes[2, col].text(0.5, 0.5, "No non-reference comparison", ha="center", transform=axes[2, col].transAxes)
            continue
        matrix = np.full((len(configs), len(xs)), np.nan)
        for i, (rec, z, rate) in enumerate(configs):
            for j, alpha in enumerate(xs):
                hits = [r for r in rs if matches(r, recommender=rec, steepness=z, r=rate, alpha=alpha)]
                if hits:
                    matrix[i, j] = 100 * hits[0]["relative_max"]
        heat = axes[2, col].imshow(np.ma.masked_invalid(matrix), norm=norm, cmap="YlOrRd", aspect="auto")
        axes[2, col].set(
            xticks=range(len(xs)),
            xticklabels=[f"{x:g}" for x in xs],
            yticks=range(len(configs)),
            yticklabels=[f"{LABELS[KERNELS.index((rec, z))]}, r={rate:g}" for rec, z, rate in configs],
            xlabel=r"Finite update size $\alpha$ (dt=1)",
            title=rule.upper() + ": maximum raw-energy gap",
        )
        axes[2, col].tick_params(axis="y", labelsize=8)
    if heat is not None:
        fig.colorbar(heat, ax=axes[2, :], label="Maximum node/edge energy gap (%)", shrink=0.85)
    decorate(axes)
    fig.suptitle(rf"Relative control and small-update similarity; $\epsilon={epsilon:g}$")
    caption = (
        "Figure 3. Fixed r=q/alpha, scaled time tau=alpha*t, and decreasing finite physical update size alpha*dt. "
        "Representative curves use Opinion4 at r=2. The summaries compare each trajectory to the smallest alpha "
        "at the same physical configuration, on their common observed window, without endpoint alignment. "
        "This is a leading small-update comparison on a fixed representation; spatial discretization is assessed separately."
    )
    finish(fig, output, "figure_3_scaling", caption, captions)


def rate_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.05, 45)
    ax.set_xticks([0.0625, 0.125, 0.5, 2, 8, 32], ["0", "1/8", "1/2", "2", "8", "32"])
    ax.axvline(0.0884, color=".7", ls=":", lw=0.8)
    ax.set_xlabel(r"Relative rate $r=q/\alpha$")


def response_curve(ax, rs, value, flag, color):
    rs = sorted(rs, key=lambda r: r["ratio"])
    xs = np.array([r["ratio"] if r["ratio"] else 0.0625 for r in rs])
    ys = np.array([r[value] for r in rs])
    good = np.array([r[flag] == "True" for r in rs])
    positive = np.array([r["ratio"] > 0 for r in rs])
    ax.plot(xs[positive], np.where(good[positive], ys[positive], np.nan), color=color, lw=1.7)
    for valid, fill in ((True, color), (False, "white")):
        take = (good == valid) & np.isfinite(ys)
        for isolated, marker in ((False, "o"), (True, "s")):
            pick = take & (positive != isolated)
            ax.plot(xs[pick], ys[pick], marker, mfc=fill, mec=color, ms=4)


def recommendation(data, epsilon, output, captions):
    rows = data.rows("recommendation.csv", epsilon)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), layout="constrained")
    for col, rule in enumerate(RULES):
        for i, (rec, z) in enumerate(KERNELS):
            rs = [r for r in rows if matches(r, dynamics=rule, recommender=rec, steepness=z)]
            response_curve(axes[0, col], rs, "delta", "confirmed", COLORS[i])
            if i:
                response_curve(axes[1, col], rs, "shift", "shift_confirmed", COLORS[i])
        axes[0, col].set_title(rule.upper())
        for ax in axes[:, col]:
            rate_axis(ax)
            ax.axhline(0, color=".3", lw=0.8)
        axes[0, col].set_ylabel(r"$\Delta=\log(A_E/A_\rho)$")
        axes[1, col].set_ylabel(r"$\Delta_a-\Delta_{\mathrm{Random}}$")
    decorate(axes)
    fig.legend(handles=kernel_legend(), loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle(rf"Recommendation changes relative node and edge responses; $\epsilon={epsilon:g}$")
    caption = (
        "Figure 4. Areas integrate normalized remaining energies over the observed window using each curve's own "
        "endpoint reference. Hollow points mark unconfirmed terminal plateaus; lines break at those points. "
        "The q=0 boundary is shown separately at the left. The log ratio and its Random contrast compare these "
        "finite-window observables; their sign changes do not establish infinite-time relaxation or formation order."
    )
    finish(fig, output, "figure_4_recommendation", caption, captions)


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
    fig, axes = plt.subplots(3, 2, figsize=(11.8, 9.5), layout="constrained")
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
    decorate(axes)
    axes[0, 0].set_title("Whole-initial bootstrap (exploratory)")
    axes[0, 1].set_title("Whole-initial bootstrap (exploratory)")
    for ax in axes[1:].flat:
        ax.set_title("LOIO sensitivity range (not 95% CI)", fontsize=10)
    fig.legend(handles=condition_legend(), loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(rf"Collective organization and variation across realizations; $\epsilon={epsilon:g}$, $\tau={at:g}$")
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
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
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
        fig.suptitle(rf"Absolute finite-window responses; $\epsilon={epsilon:g}$")
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
    nodes = data.rows("representative_nodes.csv", epsilon)
    edges = data.rows("representative_edges.csv.gz", epsilon)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for source, color in (("micro", "k"), ("pair", COLORS[2])):
        rs = sorted([r for r in nodes if r["source"] == source], key=lambda r: r["bin"])
        axes[0].plot([r["x"] for r in rs], [r["density"] for r in rs], color=color, label=source)
    axes[0].set(xlabel="Opinion", ylabel="Node density")
    axes[0].legend(frameon=False)
    vmax = max(r["density"] for r in edges)
    for ax, source in zip(axes[1:], ("micro", "pair")):
        rs = [r for r in edges if r["source"] == source]
        bins = int(rs[0]["bins"])
        matrix = np.zeros((bins, bins))
        for r in rs:
            matrix[int(r["i"]), int(r["j"])] = r["density"]
        heat = ax.imshow(
            matrix.T, origin="lower", extent=(-1, 1, -1, 1), vmin=0, vmax=vmax, cmap="magma", aspect="equal"
        )
        ax.set(xlabel="Source opinion", ylabel="Target opinion", title=source)
    fig.colorbar(heat, ax=axes[1:], label="Normalized edge density", shrink=0.8)
    decorate(axes)
    at = data.plan["analysis_protocol"]["main_snapshot_tau"]
    fig.suptitle(rf"Population-design distributions; $\epsilon={epsilon:g}$, $\tau={at:g}$")
    finish(
        fig,
        output,
        "distribution_comparison",
        "Supplement: empirical node/edge fields and the measure-map prediction from the same initial state. See Figure 1 for configuration and conditional sample count.",
        captions,
    )
    rows = data.rows("population_variation.csv", epsilon)
    for n in sorted({r["population"] for r in rows}):
        fig, axes = plt.subplots(3, 2, figsize=(11.8, 9), layout="constrained")
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
        fig.legend(handles=condition_legend(), loc="outside lower center", ncol=4, frameon=False, fontsize=9)
        fig.suptitle(rf"Population N={n:g}: variation and prediction residuals; $\epsilon={epsilon:g}$")
        finish(
            fig,
            output,
            f"variation_time_N{n:g}",
            "Supplement: W, corrected I and finite-N conditional residual B² at all recorded field snapshots. Bands are LOIO ranges, not confidence intervals. MSE=W+B²; I is not another prediction-error term. Negative corrections are retained.",
            captions,
        )


def numerical_controls(data, epsilon, output, captions):
    rows = data.rows("numerics.csv", epsilon)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
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
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
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
        if "scaling" in selected:
            scaling(data, epsilon, destination, captions)
        if "recommendation" in selected:
            recommendation(data, epsilon, destination, captions)
        if "population" in selected:
            population(data, epsilon, destination, captions)
        supplement = destination if epsilon != baseline else output / "supplementary"
        supplement.mkdir(exist_ok=True)
        supplementary(data, epsilon, supplement, captions, selected)
        data.cache.clear()
    manifest = {
        "schema": 1,
        "plan_fingerprint": data.plan["plan_fingerprint"],
        "profile": data.plan["profile"],
        "research_topics": list(selected),
        "input_scope": "portable local CSV only; no raw path lookup, solver, bootstrap or fitting",
        "analysis_manifest_sha256": sha(data.root / "analysis_manifest.json"),
        "inputs": {**data.used, **data.bundle.used},
        "renderer_sources": {
            str(path): sha(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("figure_style.py"),
                Path(inspect.getfile(setup_paper_params)),
                Path(inspect.getfile(save)),
            )
        },
        "figures": [str(Path(p).relative_to(output)) + ".pdf" for p in sorted(captions)],
        "captions": {str(Path(k).relative_to(output)): v for k, v in captions.items()},
        "smoke_is_scientific_evidence": False,
        "smoke": is_smoke(data.plan["profile"]),
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(data.root / "figure_manifest.json", manifest)
    return output
