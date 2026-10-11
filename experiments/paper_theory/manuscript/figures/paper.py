"""Assemble the frozen manuscript figure/table selection from original recorded evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.numerics.distributions import rebin_matrix

from ...paths import REVIEW_INPUT, REVISION_INPUT
from ..paths import FIGURES_OUTPUT

HERE = FIGURES_OUTPUT
EXPERIMENT = REVIEW_INPUT.resolve()

COLORS = {"random": "#246A9C", "opinion_random": "#B84236", "structure_random_l0": "#238474"}
CONFIGS = (
    ("random", 1, "Random", "#246A9C"),
    ("structure_random_l0", 1, "Structure L0, 1", "#7BAA5F"),
    ("structure_random_l0", 4, "Structure L0, 4", "#238474"),
    ("opinion_random", 1, "Opinion, 1", "#DC9B35"),
    ("opinion_random", 4, "Opinion, 4", "#B84236"),
)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 6.7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "lines.linewidth": 1.3,
        "savefig.dpi": 240,
    }
)


def sha(path):
    """Return a source-file SHA256 for manuscript provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    """Read manuscript CSV evidence as string-valued field dictionaries."""
    return list(csv.DictReader(path.open()))


def val(row, key):
    """Parse a numeric evidence cell; missing, blank or literal None becomes NaN."""
    return float(row[key]) if row.get(key) not in (None, "", "None") else np.nan


def panel(ax, letter, title):
    """Set a left-aligned panel title with its manuscript letter."""
    ax.set_title(f"({letter}) {title}", loc="left", pad=5)


def save(fig, name):
    """Save PDF/PNG copies under HERE/figures and close fig; the directory must exist."""
    for ext in ("pdf", "png"):
        fig.savefig(HERE / "figures" / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def table(name, data):
    """Export heterogeneous manuscript rows under HERE/tables; the directory must exist."""
    fields = sorted(set().union(*(r.keys() for r in data)))
    with (HERE / "tables" / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(data)


def physical(row):
    """Return the dynamics/recommender/physical/grid comparison tuple, converting numeric CSV cells to floats."""
    return tuple(
        row[k] if k in ("dynamics", "recommender") else float(row[k])
        for k in ("dynamics", "recommender", "zeta", "alpha", "ratio", "epsilon", "b")
    )


def main():
    """Render manuscript figures and provenance using complete external frozen runs and review tables; small summaries cannot replace raw trajectories."""
    parser = argparse.ArgumentParser()
    global HERE, EXPERIMENT
    parser.add_argument("--run-root", type=Path, default=REVISION_INPUT.resolve())
    parser.add_argument("--review-root", type=Path, default=REVIEW_INPUT.resolve())
    parser.add_argument("--output", type=Path, default=FIGURES_OUTPUT)
    args = parser.parse_args()
    HERE, EXPERIMENT = (args.output.expanduser().resolve(), args.review_root.expanduser().resolve())
    for name in ("figures", "tables"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    runs = {
        k: args.run_root.expanduser().resolve() / label
        for k, label in (("core", "revision_core_v3"), ("followup", "revision_followup_v1"))
    }
    plans = {k: json.loads((root / "plan.json").read_text()) for k, root in runs.items()}
    status = {
        k: {r["id"]: r for r in json.loads((root / "status.json").read_text())["records"]} for k, root in runs.items()
    }
    provenance = {
        "purpose": "manuscript figure assembly from frozen data; no solver or fitting",
        "plan_fingerprints": {k: p["plan_fingerprint"] for k, p in plans.items()},
        "raw_sha256": {},
        "table_sha256": {},
        "figure_cases": {},
    }

    def load(k, job):
        p = runs[k] / job["family"] / "raw" / (job["id"] + ".json.gz")
        digest = sha(p)
        assert digest == status[k][job["id"]]["sha256"]
        record = json.loads(gzip.decompress(p.read_bytes()))
        assert record["job"] == job
        provenance["raw_sha256"][k + "/" + job["id"]] = digest
        return record["result"]

    def read(name, followup=True):
        p = EXPERIMENT / ("review_followup_20261007" if followup else "review_20261007") / name
        provenance["table_sha256"][str(p.relative_to(EXPERIMENT))] = sha(p)
        return rows(p)

    estimates = read("conditional_clock_prediction.csv")
    contrasts = read("recommendation_contrasts.csv")
    # Conditional energy curves: the same preselected initial graph, eight processes.
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.7), layout="constrained")
    selected = []
    for i, dynamics in enumerate(("hk", "deffuant")):
        ax = axes[0, i]
        for rec in COLORS:
            jobs = [
                j
                for j in plans["followup"]["jobs"]
                if j["meta"].get("role") == "clock_ensemble"
                and j["meta"]["dynamics"] == dynamics
                and j["meta"]["initial_replicate"] == 0
                and j["meta"]["recommender"] == rec
            ]
            assert len(jobs) == 8
            curves = [load("followup", j) for j in jobs]
            selected.extend(j["id"] for j in jobs)
            t = np.asarray([f["time"] for f in curves[0]["frames"]]) * 0.02
            for channel, style in (("node", "-"), ("edge", "--")):
                micro = np.mean([[f["exact_" + channel + "_energy"] for f in r["frames"]] for r in curves], axis=0)
                pair = np.asarray([f[channel + "_energy"] for f in curves[0]["pair"]["frames"]])
                ax.plot(t, micro / micro[0], color=COLORS[rec], ls=style)
                ax.plot(t[::20], pair[::20] / pair[0], color=COLORS[rec], ls="none", marker="x", ms=3.2, mew=0.7)
        panel(ax, "ab"[i], ("HK" if i == 0 else "Deffuant") + ": fixed initial 0")
        ax.set(
            xlabel=r"$\tau=\alpha t$",
            ylabel=r"$\mathcal{U}_X(\tau)/\mathcal{U}_X(0)$",
            xlim=(0, 16),
            ylim=(-0.02, 1.04),
        )
    handles = [
        plt.Line2D(
            [],
            [],
            color=c,
            label="Structure4 / L0"
            if rec == "structure_random_l0"
            else "Opinion4"
            if rec == "opinion_random"
            else "Random",
        )
        for rec, z, _label, c in CONFIGS
        if z == 4 or rec == "random"
    ]
    handles += [
        plt.Line2D([], [], color=".35", ls="-", label="Node"),
        plt.Line2D([], [], color=".35", ls="--", label="Edge"),
        plt.Line2D([], [], color=".35", ls="none", marker="x", label="Pair"),
    ]
    fig.legend(handles=handles, ncol=6, loc="outside upper center", handlelength=1.4)
    for c, channel in enumerate(("node", "edge")):
        ax = axes[1, c]
        rs = [r for r in contrasts if r["quantity"] == channel]
        for i, r in enumerate(rs):
            point = val(r, "micro_factor")
            ax.errorbar(
                point,
                i,
                xerr=[[point - val(r, "lower")], [val(r, "upper") - point]],
                fmt="o",
                color=COLORS["opinion_random"],
                capsize=2,
                ms=3.8,
            )
            ax.plot(val(r, "pair_factor"), i, "x", color=".15", ms=5)
        ax.set_yticks(range(4), [f"{'HK' if r['dynamics'] == 'hk' else 'D'}, initial {r['initial']}" for r in rs])
        ax.invert_yaxis()
        ax.set(xlabel=r"$T_{50}^{X,\mathrm{Opinion4}}/T_{50}^{X,\mathrm{Random}}$", ylim=(3.6, -0.6))
        panel(ax, "cd"[c], channel.capitalize() + " delay factor")
        ax.axvline(1, color=".7", lw=0.7)
        if c == 0:
            ax.set_xlim(2.85, 3.95)
        else:
            ax.set_xlim(0.99, 1.26)
        ax.grid(axis="x", alpha=0.15)
    save(fig, "micro_two_channel_response")
    provenance["figure_cases"]["micro_two_channel_response"] = {
        "raw_ids": selected,
        "curves_initial": 0,
        "contrast_initials": [0, 1],
        "processes_per_initial": 8,
        "window_tau": 30,
        "curve_display_tau": [0, 16],
        "intervals": "4000 paired process-seed bootstrap, conditional on initial graph",
    }

    # Preserve original confirmation provenance while replacing exact extended cases.
    clocks = read("clocks_reviewed.csv", False)
    extensions = read("extended_platforms.csv")
    extension_index = {physical(r): r for r in extensions}
    display = []
    for original in clocks:
        if original["scan"] != "main":
            continue
        r = dict(original)
        if physical(r) in extension_index:
            r = dict(extension_index[physical(r)])
            r["confirmation_source"] = "new frozen extension protocol"
            r["plot_confirmed"] = r["energy_confirmed"]
            r["plot_joint"] = r["joint_confirmed"]
        else:
            r["plot_confirmed"] = r["energy_only_final_pass"]
            r["plot_joint"] = r["primary_joint_plateau"]
            r["confirmation_source"] = "original joint" if r["plot_joint"] == "True" else "post hoc core energy check"
            if r["plot_confirmed"] == "True":
                r["delta50"] = r["finite_delta50"]
        display.append(r)
    table("mesoscopic_clock_display.csv", display)
    fig, axes = plt.subplots(2, 3, figsize=(6.95, 4.1), layout="constrained")
    for i, d in enumerate(("hk", "deffuant")):
        for rec, z, label, color in CONFIGS:
            rs = sorted(
                [
                    r
                    for r in display
                    if r["dynamics"] == d and r["recommender"] == rec and val(r, "zeta") == z and val(r, "ratio") > 0
                ],
                key=lambda r: val(r, "ratio"),
            )
            for c, key in enumerate(("rho_tau50", "edge_tau50", "delta50")):
                xs = [val(r, "ratio") for r in rs]
                ys = [val(r, key) if r["plot_confirmed"] == "True" else np.nan for r in rs]
                axes[i, c].plot(xs, ys, color=color, ls="-" if z == 1 else "--", label=label)
                for r, x, y in zip(rs, xs, ys):
                    if np.isfinite(y):
                        axes[i, c].plot(
                            x, y, "o", ms=3.2, color=color, mfc=color if r["plot_joint"] == "True" else "white", mew=0.8
                        )
        for c, title in enumerate(("Node", "Edge", "Relative order")):
            ax = axes[i, c]
            panel(ax, "abcdef"[3 * i + c], ("HK" if i == 0 else "D") + ": " + title)
            ax.set(xscale="log", xlabel=r"$r=q/\alpha$", xlim=(0.10, 38))
            if c < 2:
                ax.set_yscale("log")
                ax.set_ylabel(r"$\alpha T_{50}^{" + (r"\rho" if c == 0 else "E") + "}$")
            else:
                ax.axhline(0, color=".5", lw=0.7)
                ax.set_ylabel(r"$\Delta_{50}=\log(T_{50}^E/T_{50}^{\rho})$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="outside upper center", handlelength=1.8)
    save(fig, "mesoscopic_recommendation_clocks")
    cross = read("crossing_ranges.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.15), layout="constrained")
    for ax, z in zip(axes, (1, 4)):
        for d, color, marker in (("hk", "#246A9C", "o"), ("deffuant", "#B84236", "s")):
            rs = sorted(
                [r for r in cross if r["dynamics"] == d and val(r, "zeta") == z], key=lambda r: val(r, "progress")
            )
            x = [val(r, "progress") / 100 for r in rs]
            lo, hi = (
                np.array([val(r, "interpolation_min") for r in rs]),
                np.array([val(r, "interpolation_max") for r in rs]),
            )
            ax.plot(x, (lo + hi) / 2, marker + "-", ms=3.5, color=color, label="HK" if d == "hk" else "Deffuant")
            ax.fill_between(x, lo, hi, alpha=0.18, color=color)
        panel(ax, "a" if z == 1 else "b", rf"Opinion, $\zeta={z}$")
        ax.set(xlabel="Release fraction p", ylabel=r"Interpolated crossing $r_*(p)$", xticks=[0.1, 0.5, 0.9])
        ax.grid(alpha=0.15)
    axes[0].legend()
    save(fig, "progress_dependent_crossings")

    # Finite-step budget and snapshot use the same HK Opinion4 state; late tail is separate.
    job = next(j for j in plans["core"]["jobs"] if j["id"] == "abaf1f5ae182bc5da198")
    record = load("core", job)
    frames = record["frames"]
    snapshot = next(f for f in frames if f.get("rho") and abs(f["time"] * 0.02 - 4) < 1e-10)
    axis = np.asarray(record["axis"])
    rho = np.asarray(snapshot["rho"])
    w = 0.5 * np.minimum((axis[:, None] - axis[None, :]) ** 2, 0.45**2)
    np.testing.assert_allclose(w @ rho, snapshot["node_potential"], atol=1e-13)
    edge = np.asarray(snapshot["edge"]).reshape(161, 161)
    phi_edge = np.divide((w * edge).sum(axis=1), 15 * rho, out=np.full(161, np.nan), where=rho > 0)
    stored = np.asarray([np.nan if v is None else v for v in snapshot["edge_potential"]])
    stored_support = np.isfinite(stored)
    np.testing.assert_allclose(phi_edge[stored_support], stored[stored_support], atol=1e-12)
    assert abs(snapshot["node_energy"] - 0.5 * rho @ w @ rho) < 1e-12
    assert abs(snapshot["edge_energy"] - (w * edge).sum() / 15) < 1e-12
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.35), layout="constrained")
    ax = axes[0, 0]
    t = np.asarray([f["time"] * 0.02 for f in frames])
    initial_energy = frames[0]["edge_energy"]
    for key, label, color in (
        ("budget_motion", "Endpoint motion", "#DC9B35"),
        ("budget_rewire", "Frozen-opinion rewiring", "#238474"),
    ):
        ax.plot(t, np.asarray([f[key] for f in frames]) / initial_energy, label=label, color=color)
    ax.plot(
        t,
        np.asarray([f["edge_energy"] - initial_energy for f in frames]) / initial_energy,
        color=".2",
        ls="--",
        label="Total edge change",
    )
    ax.set(xlabel=r"$\tau=\alpha t$", ylabel=r"Cumulative change / $\mathcal{U}_E(0)$", xlim=(0, 24))
    panel(ax, "a", "HK Opinion4, r=2: channel budget")
    ax.legend(loc="center right", fontsize=6.1)
    ax = axes[0, 1]
    ax.plot(axis, rho * 161 / 2, color=".3", label=r"$\rho$", lw=1.1)
    ax.set(xlabel="Opinion x", ylabel=r"Density $\rho(x)$", xlim=(-1, 1))
    twin = ax.twinx()
    twin.spines["right"].set_visible(True)
    occupied = rho > 1e-10
    twin.plot(axis, snapshot["node_potential"], color="#246A9C", label=r"$\Phi_\rho$")
    twin.plot(axis[occupied], phi_edge[occupied], color="#B84236", ls="--", label=r"$\Phi_E$")
    twin.set_ylabel(r"Landscape / $\mathcal{E}_\star$")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, ncol=3, fontsize=6.2, loc="upper center")
    panel(ax, "b", r"Same state at $\tau=4$")
    ax = axes[1, 0]
    gradient = np.sum((axis[:, None] - axis[None, :]) * ((np.abs(axis[:, None] - axis[None, :]) < 0.45) * rho), axis=1)
    ax.plot(
        axis[occupied], np.asarray(snapshot["velocity"])[occupied] / 0.02, color="#238474", label=r"Actual $v_a/\alpha$"
    )
    ax.plot(
        axis[occupied], -gradient[occupied], color="#246A9C", ls="--", label=r"$-\partial_x\Phi_\rho/\mathcal{E}_\star$"
    )
    ax.axhline(0, color=".65", lw=0.7)
    ax.set(xlabel="Opinion x", ylabel="Scaled drift / energy slope", xlim=(-1, 1))
    panel(ax, "c", "Actual drift and population energy slope")
    ax.legend(fontsize=6.5)
    special = next(j for j in plans["followup"]["jobs"] if j["id"] == "53a6d717ec086590dc1d")
    late = load("followup", special)
    ax = axes[1, 1]
    for channel, color in (("node", "#246A9C"), ("edge", "#B84236")):
        t = [f["time"] * 0.02 for f in late["frames"]]
        y = np.asarray([f[channel + "_energy"] for f in late["frames"]])
        ax.plot(t, y / y[0], color=color, label=channel.capitalize())
    ax.axvline(120, color=".55", ls=":", lw=0.9)
    ax.set(xlabel=r"$\tau=\alpha t$", ylabel=r"$\mathcal{U}_X(\tau)/\mathcal{U}_X(0)$", xlim=(0, 160))
    panel(ax, "d", "HK Opinion4, r=4: mesoscopic only")
    ax.legend(fontsize=6.5)
    save(fig, "energy_budget_landscape")
    budget24 = next(f for f in frames if abs(f["time"] * 0.02 - 24) < 1e-10)
    rewiring_share = budget24["budget_rewire"] / (budget24["edge_energy"] - frames[0]["edge_energy"])
    uphill = occupied & (np.asarray(snapshot["velocity"]) / 0.02 * gradient > 1e-8)
    provenance["figure_cases"]["energy_budget_landscape"] = {
        "snapshot_id": job["id"],
        "snapshot_tau": 4,
        "late_id": special["id"],
        "late_scope": "mesoscopic only; no long-time micro counterpart",
        "edge_landscape_display_mass_threshold": 1e-10,
        "budget_convention": "rewire at frozen opinions, then endpoint motion on rewired edge law",
        "rewiring_fraction_of_edge_loss_at_tau24": float(rewiring_share),
        "uphill_rows_at_snapshot": int(uphill.sum()),
        "uphill_sign_tolerance_scaled_drift_times_energy_slope": 1e-8,
        "node_mass_on_uphill_rows": float(rho[uphill].sum()),
    }

    # Reuse the exact same sixteen micro processes at every pair resolution.
    grid = read("conditional_ensemble_grid.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.95, 2.8), layout="constrained")
    colors = ("#246A9C", "#DC9B35", "#238474", "#B84236")
    for initial, color in enumerate(colors):
        rs = sorted(
            [r for r in grid if val(r, "initial") == initial and val(r, "quadrature") == 33],
            key=lambda r: val(r, "pair_bins"),
        )
        axes[0].plot(
            [val(r, "pair_bins") for r in rs],
            [val(r, "mean_rho_w1") for r in rs],
            "o-",
            ms=3,
            color=color,
            label=f"Initial {initial}",
        )
    axes[0].set(xlabel="Pair bins B", ylabel=r"$W_1$ of conditional mean", xticks=[81, 161, 241], ylim=(0, 0.40))
    panel(axes[0], "a", "Same micro ensemble, Q=33")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=6.5,
        ncol=4,
        loc="outside upper center",
        title="Initial graph in (a)",
        title_fontsize=6.5,
        handlelength=1.5,
    )
    initials = 3
    ms = [
        j
        for j in plans["followup"]["jobs"]
        if j["family"] == "micro"
        and j["meta"]["n"] == 500
        and j["meta"]["ratio"] == 10
        and j["meta"]["b"] == 161
        and j["meta"]["quadrature"] == 33
        and j["meta"]["initial_replicate"] == initials
    ]
    assert len(ms) == 16
    samples = [load("followup", j) for j in ms]
    mean_rho = np.mean([r["frames"][-1]["rho"] for r in samples], axis=0)
    common_axis = -1 + (np.arange(81) + 0.5) * 2 / 81
    mean_mass = rebin_matrix(161, 81) @ mean_rho
    axes[1].step(common_axis, mean_mass * 81 / 2, color="#246A9C", where="mid", label="Micro mean (16)")
    times = np.asarray([f["time"] for f in samples[0]["frames"]]) * 0.02
    micro_energy = np.mean([[f["exact_node_energy"] for f in r["frames"]] for r in samples], axis=0)
    axes[2].plot(times, micro_energy / micro_energy[0], color="#246A9C", label="Micro mean")
    for b, color, style in ((81, "#DC9B35", "--"), (241, "#B84236", "-")):
        j = next(
            j
            for j in plans["followup"]["jobs"]
            if j["meta"].get("role") == "basin_grid"
            and j["meta"]["initial_replicate"] == initials
            and j["meta"]["b"] == b
            and j["meta"]["quadrature"] == 33
        )
        r = load("followup", j)["pair"]["frames"]
        mass = rebin_matrix(b, 81) @ np.asarray(r[-1]["rho"])
        axes[1].step(common_axis, mass * 81 / 2, where="mid", color=color, ls=style, label=f"Pair B={b}")
        energy = np.asarray([f["node_energy"] for f in r])
        axes[2].plot(times, energy / energy[0], color=color, ls=style, label=f"Pair B={b}")
    axes[1].set(xlabel="Opinion x", ylabel="Density", xlim=(-0.75, 0.75))
    axes[2].set(
        xlabel=r"$\tau=\alpha t$", ylabel=r"$\mathcal{U}_\rho/\mathcal{U}_\rho(0)$", xlim=(0, 30), ylim=(-0.02, 1.03)
    )
    panel(axes[1], "b", r"Initial 3: density at $\tau=30$")
    panel(axes[2], "c", "Initial 3: node energy path")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=6.5,
        ncol=3,
        loc="outside lower center",
        title="Curves in (b,c)",
        title_fontsize=6.5,
        handlelength=1.5,
    )
    save(fig, "prediction_boundaries")
    provenance["figure_cases"]["prediction_boundaries"] = {
        "initials": [0, 1, 2, 3],
        "processes": 16,
        "common_comparison_bins": 81,
        "representative_initial": 3,
        "window_tau": 30,
        "pair_quadrature": 33,
        "raw_ids": [j["id"] for j in ms],
    }
    table("conditional_clock_predictions.csv", estimates)
    clock_index = {(r["dynamics"], int(r["initial"]), r["recommender"], r["channel"]): r for r in estimates}
    clock_table = r"""\begin{table*}[t]
\centering
\small
\caption{Conditional half-times in scaled units $\alpha T_{50}$. Each micro value is the clock of an eight-process mean curve, with a 95\% process-bootstrap interval conditional on the initial graph. All use $r=2$, $N=500$, $\epsilon=0.45$, $\alpha=0.02$, $B=161$, quadrature 33, and the finite window $\tau=30$. Structural micro recommendation is compared with the L0 pair proxy.}
\label{tab:supp-conditional-clocks}
\begin{tabular}{@{}lllr rlr@{}}
\toprule
Dynamics & Initial & Recommendation & Micro node [95\% CI] & Pair node & Micro edge [95\% CI] & Pair edge \\
\midrule
"""
    for dynamics in ("hk", "deffuant"):
        for initial in (0, 1):
            for rec, label in (
                ("random", "Random"),
                ("opinion_random", "Opinion4"),
                ("structure_random_l0", "Structure4/L0"),
            ):
                cells = ["HK" if dynamics == "hk" else "D", str(initial), label]
                for channel in ("node", "edge"):
                    r = clock_index[dynamics, initial, rec, channel]
                    cells += [
                        f"{val(r, 'micro_tau50'):.3f} [{val(r, 'lower'):.3f}, {val(r, 'upper'):.3f}]",
                        f"{val(r, 'pair_tau50'):.3f}",
                    ]
                clock_table += " & ".join(cells) + r" \\" + "\n"
    clock_table += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    (HERE / "tables" / "conditional_clocks.tex").write_text(clock_table)
    table("recommendation_contrasts.csv", contrasts)
    table("conditional_ensemble_grid.csv", grid)
    table("crossing_ranges.csv", cross)
    continuous = read("micro_continuous_energy.csv", False)
    primary_path = runs["core"] / "micro" / "analysis" / "trajectory_errors.csv"
    primary = {r["id"]: r for r in rows(primary_path)}
    provenance["table_sha256"][str(primary_path)] = sha(primary_path)
    core_errors = []
    for ratio in (0.25, 2, 10):
        rs = [r for r in continuous if r["tier"] == "core" and val(r, "ratio") == ratio]
        assert len(rs) == 40
        core_errors.append(
            {
                "ratio": ratio,
                "cases": 40,
                "node_w1_median": float(np.median([val(primary[r["id"]], "node_w1") for r in rs])),
                "continuous_node_gap_median": float(np.median([val(r, "exact_node_gap") for r in rs])),
                "continuous_edge_gap_median": float(np.median([val(r, "exact_edge_gap") for r in rs])),
                "continuous_node_gap_max": max(val(r, "exact_node_gap") for r in rs),
                "scope": "unscreened core, same finite window, continuous microscopic energies",
            }
        )
    table("core_prediction_errors.csv", core_errors)
    provenance["script_sha256"] = sha(Path(__file__))
    provenance["figure_pdf_sha256"] = {p.name: sha(p) for p in (HERE / "figures").glob("*.pdf")}
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {"raw_files_verified": len(provenance["raw_sha256"]), "figure_pdfs": list(provenance["figure_pdf_sha256"])}
        )
    )


if __name__ == "__main__":
    main()
