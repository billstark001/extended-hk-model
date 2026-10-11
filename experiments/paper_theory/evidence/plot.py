"""Metadata-selected standalone figures from the fresh evidence tables/records."""

import json
from collections import defaultdict

import numpy as np

from common.plotting.batch import plotting, savefig

from ..data import read_csv
from ..runtime.measure_analysis import cached_cases


def groups(rows, fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in fields)].append(row)
    return result


def kernel(row):
    names = {
        "random": "Random",
        "opinion_random": "Opinion (weighted)",
        "structure_random_l0": "Structure L0 (weighted)",
    }
    name = names[row["recommender"]]
    return name if row["recommender"] == "random" else f"{name} z={float(row['steepness']):g}"


def number(row, name):
    value = row.get(name)
    return float(value) if value not in (None, "", "None") else np.nan


def band(ax, rows, x, metric, label):
    rows = sorted(rows, key=lambda r: number(r, x))
    xx = [number(r, x) for r in rows]
    yy = [number(r, metric + "_mean") for r in rows]
    (line,) = ax.plot(xx, yy, "o-", label=label)
    ax.fill_between(
        xx,
        [number(r, metric + "_lower") for r in rows],
        [number(r, metric + "_upper") for r in rows],
        color=line.get_color(),
        alpha=0.15,
    )
    return line


def plot(directory):
    plan = json.loads((directory / "plan.json").read_text())
    summary = json.loads((directory / "report.json").read_text())
    if summary["plan_fingerprint"] != plan["plan_fingerprint"]:
        raise ValueError("report/plan mismatch; analyze first")
    plt = plotting()
    figures = directory / "figures"
    written = []
    rows = read_csv(directory / "tables" / "mean_release.csv")
    horizon = max(number(r, "horizon_tau") for r in rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    for i, rule in enumerate(("hk", "deffuant")):
        for j, ch in enumerate(("node", "edge")):
            selected = [
                r
                for r in rows
                if r["dynamics"] == rule
                and r["channel"] == ch
                and "effects" in r["roles"]
                and np.isclose(number(r, "horizon_tau"), horizon)
                and np.isclose(number(r, "epsilon"), 0.4)
            ]
            for _, members in groups(selected, ("recommender", "steepness")).items():
                members.sort(key=lambda r: number(r, "ratio"))
                axes[i, j].plot(
                    [number(r, "ratio") for r in members],
                    [number(r, "mean_tau") for r in members],
                    "o-",
                    label=kernel(members[0]),
                )
            axes[i, j].set(xlabel="r = q/alpha", ylabel=f"{rule}: {ch} restricted mean (tau)", xscale="symlog")
            axes[i, j].legend(fontsize=6)
    fig.suptitle("Observed window; endpoint estimates require plateau diagnostics")
    savefig(fig, figures, "mean_release")
    written.append("mean_release")
    cases = list(cached_cases(directory / "measure"))
    tasks = {t["id"]: t for t in plan["tasks"]}
    kernels = sorted({(c["recommender"], c["steepness"]) for c in cases})
    for channel, name in (("edge", "scaling_collapse"), ("node", "scaling_collapse_node")):
        fig, axes = plt.subplots(2, len(kernels), figsize=(3.4 * len(kernels), 6), squeeze=False)
        for i, rule in enumerate(("hk", "deffuant")):
            for j, (rec, zeta) in enumerate(kernels):
                selected = sorted(
                    [
                        c
                        for c in cases
                        if c["dynamics"] == rule
                        and c["recommender"] == rec
                        and c["steepness"] == zeta
                        and "scaling" in tasks[c["source"].removesuffix(".npz")]["roles"]
                        and np.isclose(c["ratio"], 2.0)
                    ],
                    key=lambda c: c["alpha"],
                )
                for c in selected:
                    axes[i, j].plot(
                        c["tau"], c[channel + "_energy"], label=f"alpha={c['alpha']:g}, q={c['alpha'] * c['ratio']:g}"
                    )
                axes[i, j].set(
                    title=f"{rule}: {rec} z={zeta:g}", xlabel="tau = alpha*t", ylabel=f"raw {channel} energy"
                )
                axes[i, j].legend(fontsize=6)
        fig.suptitle("Fixed r=2 within each panel: raw curves; state/errors in tables")
        savefig(fig, figures, name)
    written.append("scaling_collapse")
    rows = read_csv(directory / "tables" / "conditional_means.csv")
    n = min(number(r, "population") for r in rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    for i, rule in enumerate(("hk", "deffuant")):
        selected = [
            r
            for r in rows
            if r["dynamics"] == rule
            and r["plateau_reference"] == "own_endpoint"
            and number(r, "initial") == 0
            and number(r, "population") == n
            and number(r, "bins") == (21 if plan["profile"] == "smoke" else 81)
            and number(r, "quadrature") == (3 if plan["profile"] == "smoke" else 9)
        ]
        for _, members in groups(selected, ("recommender", "steepness")).items():
            for j, ch in enumerate(("node", "edge")):
                line = band(axes[i, j], members, "ratio", ch, kernel(members[0]) + " micro")
                ordered = sorted(members, key=lambda r: number(r, "ratio"))
                axes[i, j].plot(
                    [number(r, "ratio") for r in ordered],
                    [number(r, "pair_" + ch + "_mean") for r in ordered],
                    "--x",
                    color=line.get_color(),
                )
        for j, ch in enumerate(("node", "edge")):
            axes[i, j].set(xlabel="r = q/alpha", ylabel=f"{rule}: {ch} restricted mean (tau)")
            axes[i, j].legend(fontsize=6)
    fig.suptitle("Fixed initial state: process-bootstrap bands; dashed matched pair")
    savefig(fig, figures, "matched_micro_means")
    written.append("matched_micro_means")
    rows = read_csv(directory / "tables" / "population.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    for i, rule in enumerate(("hk", "deffuant")):
        selected = [
            r
            for r in rows
            if r["dynamics"] == rule
            and np.isclose(number(r, "ratio"), 2.0)
            and number(r, "bins") == (21 if plan["profile"] == "smoke" else 81)
            and number(r, "quadrature") == (3 if plan["profile"] == "smoke" else 9)
        ]
        for _, members in groups(selected, ("recommender", "steepness")).items():
            for j, metric in enumerate(("micro_variance", "micro_discordant", "node_w1")):
                band(axes[i, j], members, "population", metric, kernel(members[0]))
        for j, metric in enumerate(("micro_variance", "micro_discordant", "node_w1")):
            axes[i, j].set(
                xlabel="N; k=15, R=10 fixed",
                ylabel=f"{rule}: "
                + {
                    "micro_variance": "opinion variance",
                    "micro_discordant": "discordant edge fraction",
                    "node_w1": "micro/pair node W1",
                }[metric],
            )
            axes[i, j].legend(fontsize=6)
    fig.suptitle("Independent initial-state/process intervals; finite-N robustness")
    savefig(fig, figures, "population_robustness")
    written.append("population_robustness")
    rows = read_csv(directory / "tables" / "numerics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    for j, ch in enumerate(("node", "edge")):
        selected = [r for r in rows if r["channel"] == ch]
        labels = [f"{r['dynamics']} {kernel(r)} B={r['bins']} Q={r['quadrature']} dt={r['dt']}" for r in selected]
        axes[0, j].plot(
            range(len(selected)), [number(r, "relative_max") for r in selected], "o", label="max raw curve gap"
        )
        axes[0, j].plot(
            range(len(selected)),
            [number(r, "mean_relative_change") for r in selected],
            "x",
            label="restricted mean change",
        )
        axes[0, j].set(xticks=range(len(labels)), xticklabels=labels, ylabel=ch + " relative discrepancy")
        axes[0, j].tick_params(axis="x", labelrotation=90, labelsize=5)
        axes[0, j].legend(fontsize=7)
    fig.suptitle("Predeclared grid, quadrature and dt controls")
    savefig(fig, figures, "numerical_controls")
    written.append("numerical_controls")
    (directory / "figures.json").write_text(
        json.dumps({"plan_fingerprint": plan["plan_fingerprint"], "figures": written}, indent=2) + "\n"
    )
    print(f"assembled {len(written)} figure groups from this run", flush=True)
    return written
