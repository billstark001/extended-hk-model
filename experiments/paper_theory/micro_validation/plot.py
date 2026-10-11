"""Microscopic/pair field, energy and conditional diagnostic panels."""

from __future__ import annotations

import numpy as np

from common.io.json import atomic_json
from common.plotting.batch import plotting, savefig


def plot_micro(items, fields, out):
    """Render field/energy discrepancy and conditional micro/pair diagnostics from validated analysis outputs."""
    if not items:
        return []
    plt = plotting()
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    selected = []
    for d in ("hk", "deffuant"):
        for recname in ("random", "opinion_random", "structure_random_l0"):
            found = [
                r
                for r in items
                if r["job"]["meta"]["dynamics"] == d
                and r["job"]["meta"]["recommender"] == recname
                and r["job"]["meta"]["selection"] == "weighted"
                and r["job"]["tier"] == "core"
            ]
            if found:
                selected.append(
                    min(
                        found,
                        key=lambda r: (
                            abs(r["job"]["meta"]["ratio"] - 2),
                            r["job"]["meta"]["replicate"],
                            -r["job"]["meta"]["zeta"],
                            r["job"]["id"],
                        ),
                    )
                )
    for record in selected:
        md = record["job"]["meta"]
        row = 0 if md["dynamics"] == "hk" else 1
        for col, name in enumerate(("node_energy", "edge_energy")):
            for route, fs, style in (
                ("micro", record["result"]["frames"], "-"),
                ("pair", record["result"]["pair"]["frames"], "--"),
            ):
                axs[row, col].plot(
                    [md["alpha"] * f["time"] for f in fs],
                    [f[name] for f in fs],
                    style,
                    label=md["recommender"] + " " + route,
                )
    for ax in axs.flat:
        ax.set_xlabel(r"$\alpha t$")
        ax.set_ylabel("interaction energy")
        ax.legend(fontsize=6)
    for i, d in enumerate(("HK", "Deffuant")):
        for j, name in enumerate(("node", "edge")):
            axs[i, j].set_title(f"{d}: {name}")
    savefig(fig, out / "figures", "matched_energy_trajectories")
    atomic_json(out / "figure_cases.json", [{"id": r["job"]["id"], "meta": r["job"]["meta"]} for r in selected])
    dist = [r for r in selected if r["job"]["meta"]["recommender"] == "opinion_random"]
    if dist:
        fig, axs = plt.subplots(len(dist), 3, figsize=(11, 3.2 * len(dist)), squeeze=False)
        for i, r in enumerate(dist):
            md = r["job"]["meta"]
            a = r["result"]["frames"][-1]
            p = r["result"]["pair"]["frames"][-1]
            x = r["result"]["pair"]["axis"]
            axs[i, 0].plot(x, a["rho"], label="micro")
            axs[i, 0].plot(x, p["rho"], "--", label="pair")
            axs[i, 0].legend()
            axs[i, 0].set_title(md["dynamics"] + ": node mass")
            axs[i, 0].set_xlabel("opinion")
            axs[i, 0].set_ylabel("cell mass")
            mats = [
                np.asarray(f["edge"]).reshape(md["b"], md["b"])
                / md.get("degree", r["job"]["payload"]["request"]["out_degree"])
                for f in (a, p)
            ]
            vmax = max(float(m.max()) for m in mats)
            for j, m in enumerate(mats, 1):
                im = axs[i, j].imshow(m.T, origin="lower", extent=(-1, 1, -1, 1), vmin=0, vmax=vmax, aspect="auto")
                axs[i, j].set_title(("micro" if j == 1 else "pair") + ": E/k")
                axs[i, j].set_xlabel("source opinion")
                axs[i, j].set_ylabel("target opinion")
            fig.colorbar(im, ax=list(axs[i, 1:]), label="cell mass")
        savefig(fig, out / "figures", "matched_final_distributions")
    return ["matched_energy_trajectories.pdf"] + (["matched_final_distributions.pdf"] if dist else [])
