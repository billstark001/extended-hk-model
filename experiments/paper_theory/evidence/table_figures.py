"""Publication plots from frozen CSV exports; never build or execute a solver."""

import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from common.io.json import atomic_json, fingerprint, sha
from common.plotting.figures import decorate, numeric_log_ticks

from ..types import FrozenMainPlan
from .figure_style import COLORS, LABELS, RULES, matches, read_table, save
from .protocol import KERNELS


def selected(directory, name, epsilon, **values):
    return [r for r in read_table(directory, name, epsilon) if matches(r, **values)]


def legend(fig, pairs=False):
    handles = [Line2D([], [], color=c, label=label, lw=2) for c, label in zip(COLORS, LABELS)]
    if pairs:
        handles += [Line2D([], [], color="k", ls=":", label="Pair prediction")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)


def endpoint_rows(directory, plan):
    p = plan["analysis_protocol"]
    return [
        r
        for r in read_table(directory, "mean_release", p["baseline_epsilon"])
        if "effects" in r["roles"].split(",") and r["is_observed_endpoint"] == "True"
    ]


def clock_groups(rows, rule):
    groups = {}
    for kernel in KERNELS:
        channels = {}
        for channel in ("node", "edge"):
            channels[channel] = {
                r["ratio"]: r
                for r in rows
                if matches(r, dynamics=rule, recommender=kernel[0], steepness=kernel[1], channel=channel)
            }
        groups[kernel] = channels
    return groups


def curve(ax, xs, ys, confirmed, color, index):
    """Keep unsupported points visible, but break lines at unconfirmed values."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    good = np.asarray(confirmed, dtype=bool) & np.isfinite(ys)
    positive = xs > 0
    ax.plot(
        xs[positive],
        np.where(good[positive], ys[positive], np.nan),
        color=color,
        lw=2.0 if index in (1, 2) else 1.25,
        ls="-" if index != 1 else "--",
        alpha=1 if index < 3 else 0.7,
    )
    for valid, fill in ((True, color), (False, "white")):
        take = positive & np.isfinite(ys) & (good == valid)
        ax.plot(xs[take], ys[take], "o", ms=4, mec=color, mfc=fill, mew=1.2)
    # q=0 has a distinct endpoint reference. Never connect it to positive r.
    take = (xs == 0) & np.isfinite(ys)
    for y, valid in zip(ys[take], good[take]):
        ax.plot(0.0625, y, "s", ms=4, mec=color, mfc=color if valid else "white")


def rate_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.05, 45)
    ax.set_xticks([0.0625, 0.125, 0.5, 2, 8, 32], ["0", "1/8", "1/2", "2", "8", "32"])
    ax.axvline(0.0884, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel(r"Relative rate $r=q/\alpha$; isolated $r=0$ at left")


def recommendation_figure(directory, output, plan):
    rows = endpoint_rows(directory, plan)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), layout="constrained")
    for col, rule in enumerate(RULES):
        groups = clock_groups(rows, rule)
        baseline = groups[KERNELS[0]]
        for i, (color, kernel) in enumerate(zip(COLORS, KERNELS)):
            g = groups[kernel]
            xs = sorted(g["node"].keys() & g["edge"].keys())
            ys, shifts, good, shift_good = [], [], [], []
            for r in xs:
                n, e = g["node"][r], g["edge"][r]
                delta = np.log(e["mean_tau"] / n["mean_tau"]) if min(e["mean_tau"], n["mean_tau"]) > 0 else np.nan
                bn, be = baseline["node"][r], baseline["edge"][r]
                base = np.log(be["mean_tau"] / bn["mean_tau"]) if min(be["mean_tau"], bn["mean_tau"]) > 0 else np.nan
                confirmed = n["plateau_confirmed"] == e["plateau_confirmed"] == "True"
                ys.append(delta)
                shifts.append(delta - base)
                good.append(confirmed)
                shift_good.append(confirmed and bn["plateau_confirmed"] == be["plateau_confirmed"] == "True")
            curve(axes[0, col], xs, ys, good, color, i)
            if i:
                curve(axes[1, col], xs, shifts, shift_good, color, i)
        for ax in axes[:, col]:
            rate_axis(ax)
            ax.axhline(0, color="0.25", lw=0.9)
        axes[0, col].set_title("HK" if rule == "hk" else "One-sided Deffuant")
        axes[0, col].set_ylabel(r"Relative clock $\Delta=\log(A_E/A_\rho)$")
        axes[1, col].set_ylabel(r"Recommendation shift $\Delta_a-\Delta_{\rm Random}$")
    decorate(axes)
    legend(fig)
    fig.suptitle(
        "Recommendation changes the relative node / edge clock\n"
        "Observed areas, own endpoint references; hollow points: unconfirmed terminal plateau",
        fontsize=11,
    )
    save(fig, output, "mean_recommendation_clocks", plan["analysis_protocol"]["baseline_epsilon"])


def absolute_figure(directory, output, plan):
    rows = endpoint_rows(directory, plan)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), layout="constrained")
    for col, rule in enumerate(RULES):
        for i, (color, kernel) in enumerate(zip(COLORS, KERNELS)):
            for row, channel in enumerate(("node", "edge")):
                rs = sorted(
                    (
                        r
                        for r in rows
                        if matches(r, dynamics=rule, recommender=kernel[0], steepness=kernel[1], channel=channel)
                    ),
                    key=lambda r: r["ratio"],
                )
                curve(
                    axes[row, col],
                    [r["ratio"] for r in rs],
                    [r["mean_tau"] for r in rs],
                    [r["plateau_confirmed"] == "True" for r in rs],
                    color,
                    i,
                )
                axes[row, col].set_yscale("log")
                numeric_log_ticks(axes[row, col])
                rate_axis(axes[row, col])
                axes[row, col].set_ylabel(f"Observed {channel} residual area")
        axes[0, col].set_title("HK" if rule == "hk" else "One-sided Deffuant")
    decorate(axes)
    legend(fig)
    fig.suptitle(
        "Absolute release areas in scaled time\nOwn endpoint references; observed windows are not extrapolated",
        fontsize=11,
    )
    save(fig, output, "absolute_release_areas", plan["analysis_protocol"]["baseline_epsilon"])


def matched_areas(directory, output, plan):
    p = plan["analysis_protocol"]
    rows = selected(
        directory,
        "conditional_means",
        p["baseline_epsilon"],
        population=p["matched_population"],
        bins=p["matched_bins"],
        quadrature=p["matched_quadrature"],
        horizon=p["maximum_horizon_tau"],
        plateau_reference="own_endpoint",
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.2), layout="constrained")
    for col, rule in enumerate(RULES):
        for row, channel in enumerate(("node", "edge")):
            ax = axes[row, col]
            for j, idx in enumerate((0, 2, 4)):
                rec, z = KERNELS[idx]
                for initial, marker in ((0, "o"), (1, "s")):
                    r = next(
                        r for r in rows if matches(r, dynamics=rule, recommender=rec, steepness=z, initial=initial)
                    )
                    x = j + (initial - 0.5) * 0.18
                    ax.vlines(x, r[channel + "_lower"], r[channel + "_upper"], color=COLORS[idx], lw=1.8)
                    ax.plot(x, r[channel + "_mean"], marker, color=COLORS[idx], ms=6)
                    ax.plot(x + 0.055, r["pair_" + channel + "_mean"], "x", color=COLORS[idx], ms=7, mew=1.8)
            ax.set(
                xticks=range(3),
                xticklabels=["Random", "Opinion4", "Structure4"],
                ylabel=f"{channel.capitalize()} residual area",
                xlim=(-0.4, 2.4),
            )
            if channel == "node" and all(r[channel + "_lower"] > 0 for r in rows):
                ax.set_yscale("log")
                numeric_log_ticks(ax)
        axes[0, col].set_title("HK" if rule == "hk" else "One-sided Deffuant")
    decorate(axes)
    fig.legend(
        handles=[
            Line2D([], [], color="0.3", marker=m, ls="", label=label)
            for m, label in [("o", "Micro: initial 0"), ("s", "Micro: initial 1"), ("x", "Pair")]
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        f"Matched process means through $\\tau={p['maximum_horizon_tau']:g}$\n"
        "Whole-process 95% bootstrap intervals; each curve uses its own endpoint",
        fontsize=11,
    )
    save(fig, output, "matched_area_validation", p["baseline_epsilon"])


def population_rows(directory, plan):
    p = plan["analysis_protocol"]
    return selected(
        directory,
        "population",
        p["baseline_epsilon"],
        bins=p["main_population_bins"],
        quadrature=p["main_population_quadrature"],
        horizon=p["population_horizon_tau"],
    )


def population_panel(ax, rows, rule, metric, label, pairs=False):
    for idx in (0, 2):
        rec, z = KERNELS[idx]
        rs = sorted(
            (r for r in rows if matches(r, dynamics=rule, recommender=rec, steepness=z)), key=lambda r: r["population"]
        )
        xs = [r["population"] for r in rs]
        ax.plot(xs, [r[metric + "_mean"] for r in rs], "o-", color=COLORS[idx], ms=5, lw=1.8)
        ax.fill_between(
            xs, [r[metric + "_lower"] for r in rs], [r[metric + "_upper"] for r in rs], color=COLORS[idx], alpha=0.12
        )
        if pairs:
            ax.plot(xs, [r[metric.replace("micro_", "pair_") + "_mean"] for r in rs], ":", color=COLORS[idx], lw=1.3)
    ax.set_xscale("log", base=2)
    ns = sorted({r["population"] for r in rows})
    ax.set_xticks(ns, [f"{n:g}" for n in ns])
    ax.set(xlabel="Population N", ylabel=label)


def population_figures(directory, output, plan):
    p = plan["analysis_protocol"]
    rows = population_rows(directory, plan)
    structure = (
        ("micro_variance", "Opinion variance"),
        ("micro_pairing_excess", "Concordance minus marginal reference"),
    )
    near_consensus = all(
        abs(r[m + bound]) < 1e-12 for r in rows for m, _ in structure for bound in ("_lower", "_upper")
    )
    if near_consensus:
        fig, ax = plt.subplots(figsize=(9, 1.8), layout="constrained")
        ax.axis("off")
        ax.text(0.5, 0.6, "Near consensus at every tested population", ha="center", fontsize=14)
        ax.text(
            0.5,
            0.25,
            "HK and Deffuant; Random and Opinion4; N="
            + ", ".join(f"{n:g}" for n in sorted({r["population"] for r in rows}))
            + "\n"
            "Variance and excess concordance: bootstrap upper bounds below $10^{-12}$",
            ha="center",
            fontsize=10,
        )
        save(fig, output, "population_structure", p["baseline_epsilon"])
    else:
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.2), layout="constrained")
        for col, rule in enumerate(RULES):
            for row, (metric, label) in enumerate(structure):
                population_panel(axes[row, col], rows, rule, metric, label, pairs=True)
            axes[0, col].set_title("HK" if rule == "hk" else "One-sided Deffuant")
        decorate(axes)
        fig.legend(
            handles=[Line2D([], [], color=COLORS[i], label=LABELS[i]) for i in (0, 2)]
            + [Line2D([], [], color="0.3", ls=":", label="Pair prediction")],
            loc="outside lower center",
            ncol=3,
            frameon=False,
        )
        fig.suptitle(
            f"Collective structure at $\\tau={p['population_horizon_tau']:g}$\nHierarchical 95% bootstrap intervals",
            fontsize=11,
        )
        save(fig, output, "population_structure", p["baseline_epsilon"])
    for name, metrics in (
        (
            "population_closure_errors",
            (("node_w1", r"Node transport distance $W_1$"), ("edge_hellinger", "Edge Hellinger distance")),
        ),
        (
            "population_release_areas",
            (
                ("node", "Node residual area"),
                ("edge", "Edge residual area"),
                ("edge_to_node", "Edge / node area ratio"),
            ),
        ),
    ):
        fig, axes = plt.subplots(len(metrics), 2, figsize=(9.4, 3 * len(metrics)), layout="constrained")
        for col, rule in enumerate(RULES):
            for row, (metric, label) in enumerate(metrics):
                population_panel(axes[row, col], rows, rule, metric, label)
            axes[0, col].set_title("HK" if rule == "hk" else "One-sided Deffuant")
        decorate(axes)
        fig.legend(
            handles=[Line2D([], [], color=COLORS[i], label=LABELS[i]) for i in (0, 2)],
            loc="outside lower center",
            ncol=2,
            frameon=False,
        )
        fig.suptitle(
            f"Population diagnostics through $\\tau={p['population_horizon_tau']:g}$\n"
            "Hierarchical 95% bootstrap intervals; areas use own endpoints",
            fontsize=11,
        )
        save(fig, output, name, p["baseline_epsilon"])


def scaling_errors(directory, output, plan):
    p = plan["analysis_protocol"]
    rows = read_table(directory, "scaling", p["baseline_epsilon"])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), layout="constrained")
    for col, rule in enumerate(RULES):
        for row, channel in enumerate(("node_energy", "edge_energy")):
            ax = axes[row, col]
            for idx, (rec, z) in enumerate(KERNELS):
                rs = [r for r in rows if matches(r, dynamics=rule, recommender=rec, steepness=z, energy=channel)]
                alphas = sorted({r["alpha"] for r in rs})
                ax.plot(
                    alphas,
                    [100 * max(r["relative_max"] for r in rs if r["alpha"] == a) for a in alphas],
                    "o-",
                    color=COLORS[idx],
                    ms=4,
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            numeric_log_ticks(ax)
            ax.set_xticks(alphas, [f"{a:g}" for a in alphas])
            ax.set(xlabel=r"Update size $\alpha$", ylabel=f"Maximum {channel.split('_')[0]} curve gap (%)")
        axes[0, col].set_title("HK" if rule == "hk" else "One-sided Deffuant")
    decorate(axes)
    legend(fig)
    ref = min(r["reference_alpha"] for r in rows)
    fig.suptitle(
        f"Raw-energy scaling errors against alpha={ref:g}\n"
        "Maximum across declared r values; normalized by reference energy scale, common observed intervals",
        fontsize=11,
    )
    save(fig, output, "scaling_errors", p["baseline_epsilon"])


def numerical_figure(directory, output, plan):
    p = plan["analysis_protocol"]
    rows = read_table(directory, "numerics", p["baseline_epsilon"])
    factors = [("bins", 81, "B=81"), ("bins", 241, "B=241"), ("quadrature", 9, "Q=9"), ("dt", 0.5, "dt=0.5")]
    # Smoke uses different mesh/quadrature controls; infer its factors from rows.
    if plan["profile"] == "main_smoke":
        factors = [("bins", 31, "B=31"), ("quadrature", 5, "Q=5"), ("dt", 0.5, "dt=0.5")]
    columns = [(rec, z, channel) for rec, z in (KERNELS[2], KERNELS[4]) for channel in ("node", "edge")]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), layout="constrained")
    norm = LogNorm(vmin=0.001, vmax=100)
    for col, rule in enumerate(RULES):
        for row, metric in enumerate(("relative_max", "mean_relative_change")):
            values = np.full((len(factors), len(columns)), np.nan)
            for i, (field, value, _) in enumerate(factors):
                for j, (rec, z, channel) in enumerate(columns):
                    rs = [
                        r
                        for r in rows
                        if matches(r, dynamics=rule, recommender=rec, steepness=z, channel=channel, **{field: value})
                    ]
                    if len(rs) == 1:
                        values[i, j] = 100 * rs[0][metric]
            ax = axes[row, col]
            heat = ax.imshow(np.ma.masked_invalid(np.maximum(values, 0.001)), cmap="YlOrRd", norm=norm, aspect="auto")
            ax.set(
                xticks=range(4),
                xticklabels=["Opinion4\nnode", "Opinion4\nedge", "Structure4\nnode", "Structure4\nedge"],
                yticks=range(len(factors)),
                yticklabels=[f[2] for f in factors],
            )
            for (i, j), value in np.ndenumerate(values):
                if np.isfinite(value):
                    ax.text(
                        j,
                        i,
                        f"{value:.2g}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if value > 3 else "0.15",
                    )
            ax.set_title(
                ("HK" if rule == "hk" else "One-sided Deffuant")
                + "\n"
                + ("Raw-energy maximum gap (%)" if row == 0 else "Residual-area change (%)")
            )
    fig.colorbar(heat, ax=axes, label="Percent (logarithmic colour scale)", shrink=0.8)
    baseline = p["numerical_baseline"]
    fig.suptitle(
        f"Single-factor numerical controls at r=2\nBaseline B={baseline['bins']}, Q={baseline['quadrature']}, dt=1; final fields and observed endpoints may differ",
        fontsize=11,
    )
    save(fig, output, "numerical_controls", p["baseline_epsilon"])


def plot_tables(directory, output=None):
    directory = Path(directory)
    plan = cast(FrozenMainPlan, json.loads((directory / "plan.json").read_text()))
    if fingerprint({k: v for k, v in plan.items() if k != "plan_fingerprint"}) != plan["plan_fingerprint"]:
        raise ValueError("frozen plan hash mismatch")
    report = json.loads((directory / "report.json").read_text())
    if report["missing"] or report["completed"] != report["planned"]:
        raise ValueError("publication plots require complete declared coverage")
    output = Path(output) if output else directory / "figures_csv"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "savefig.facecolor": "white",
            "axes.grid": False,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    for epsilon in plan["analysis_protocol"]["confidence_radii"]:
        view = dict(plan, analysis_protocol=dict(plan["analysis_protocol"], baseline_epsilon=epsilon))
        target = output if epsilon == plan["analysis_protocol"]["baseline_epsilon"] else output / f"epsilon_{epsilon:g}"
        target.mkdir(exist_ok=True)
        recommendation_figure(directory, target, view)
        absolute_figure(directory, target, view)
        matched_areas(directory, target, view)
        population_figures(directory, target, view)
        scaling_errors(directory, target, view)
        numerical_figure(directory, target, view)
    atomic_json(
        output / "manifest.json",
        {
            "schema": 1,
            "plan_fingerprint": plan["plan_fingerprint"],
            "profile": plan["profile"],
            "input_scope": "frozen local CSV exports only; no simulation or raw-data migration",
            "input_sha256": {
                str(p.relative_to(directory)): sha(p) for p in sorted((directory / "tables").glob("*.csv"))
            },
            "figures": [str(p.relative_to(output)) for p in sorted(output.rglob("*.pdf"))],
            "raw_curve_figures": "retained original exports; redesign requires raw trajectories",
            "clock_scope": "unclipped observed-window residual areas, own endpoint references",
        },
    )
    print(f"CSV figures saved to {output}", flush=True)
    return output
