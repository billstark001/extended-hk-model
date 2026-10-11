"""Minimal observable comparison, retaining the original KWW curve model."""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from theory.release.curves import log_rate_crossings, norm, quantiles
from theory.release.fitting import fit_clock_ratio

from .analyze.curves import cases, key
from .analyze.tables import write_csv
from .artifact_paths import DATA, FIGURES, INPUT_RAW, SOURCE, ensure_output_dirs

COLOR = {"hk": "#2166ac", "deffuant": "#c42935"}


def main():
    """Compare observed and fitted release quantiles/curve reconstruction, preserving historical KWW normalization."""
    ensure_output_dirs()
    cs = list(cases())
    rows = []
    old = {}
    with (SOURCE / "edge_timescale_fits.csv").open() as f:
        for d in csv.DictReader(f):
            old[
                (d["dynamics"], round(float(d["alpha"]), 12), float(d["epsilon"]), round(float(d["q_over_alpha"]), 9))
            ] = d
    for c in cs:
        if c["r"] <= 0:
            continue
        base = {k: c[k] for k in ("dynamics", "epsilon", "alpha", "r", "converged")}
        te = quantiles(c["t"], c["se"], [0.1, 0.5, 0.9])
        tn = quantiles(c["t"], c["sn"], [0.1, 0.5, 0.9])
        p = old[key(c)]
        w, k, b = [float(p[n]) for n in ("weight", "kappa", "beta")]
        prediction = w * np.exp(-((k * c["r"] * c["t"]) ** b)) + (1 - w) * c["sn"]
        tf = quantiles(c["t"], prediction, [0.5])[0]
        rows.append(
            dict(
                **base,
                TE10=te[0],
                TE50=te[1],
                TE90=te[2],
                TN10=tn[0],
                TN50=tn[1],
                TN90=tn[2],
                log_ratio=float(np.log(te[1] / tn[1])),
                log_ratio10=float(np.log(te[0] / tn[0])),
                log_ratio90=float(np.log(te[2] / tn[2])),
                fit_TE50=tf,
                fit_log_ratio=float(np.log(tf / tn[1])),
            )
        )
    write_csv(DATA / "clock_comparison.csv", rows)
    extra = []
    for path in (INPUT_RAW).glob("*.npz"):
        z = np.load(path)
        t = z["tau"]
        se, _, _ = norm(z["edge_energy"])
        sn, _, _ = norm(z["node_energy"])
        te = quantiles(t, se, [0.5])[0]
        tn = quantiles(t, sn, [0.5])[0]
        extra.append(
            dict(
                dynamics=str(z["dynamics"]),
                epsilon=float(z["epsilon"]),
                r=float(z["r"]),
                TE50=te,
                TN50=tn,
                log_ratio=float(np.log(te / tn)),
            )
        )
    write_csv(DATA / "new_clock_comparison.csv", extra)
    crossings = []
    for dyn in ("hk", "deffuant"):
        for eps in (0.2, 0.4, 0.8):
            rr = sorted(set(d["r"] for d in rows if d["dynamics"] == dyn and d["epsilon"] == eps and d["r"] <= 100))
            for name in ("log_ratio10", "log_ratio", "log_ratio90"):
                yy = np.array(
                    [
                        np.median(
                            [d[name] for d in rows if d["dynamics"] == dyn and d["epsilon"] == eps and d["r"] == r]
                        )
                        for r in rr
                    ]
                )
                roots = log_rate_crossings(np.log(rr), yy)
                crossings.append(
                    dict(
                        dynamics=dyn,
                        epsilon=eps,
                        progress={"log_ratio10": 0.1, "log_ratio": 0.5, "log_ratio90": 0.9}[name],
                        roots=roots,
                        source="five-alpha median original grid",
                    )
                )
            # New simulations resolve the half-time crossing at alpha=.01.
            ss = sorted(
                [d for d in extra if d["dynamics"] == dyn and d["epsilon"] == eps and d["r"] < 3], key=lambda d: d["r"]
            )
            if ss:
                xx = np.log([d["r"] for d in ss])
                yy = np.array([d["log_ratio"] for d in ss])
                roots = log_rate_crossings(xx, yy)
                crossings.append(
                    dict(dynamics=dyn, epsilon=eps, progress=0.5, roots=roots, source="new alpha=.01 grid")
                )
    (DATA / "clock_crossings.json").write_text(json.dumps(crossings, indent=2))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 170})
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.5), sharex=True)
    for j, eps in enumerate((0.2, 0.4, 0.8)):
        axs[0, j].set_title(r"$\epsilon=" + str(eps) + "$")
        for dyn in ("hk", "deffuant"):
            group = [d for d in rows if d["dynamics"] == dyn and d["epsilon"] == eps and d["r"] <= 100]
            rr = np.array(sorted(set(d["r"] for d in group)))
            ls = "-" if dyn == "hk" else "--"
            mk = "o" if dyn == "hk" else "^"
            for field, col, label in [("TE50", "#177e89", "Edge"), ("TN50", "#d17b0f", "Node")]:
                for a in sorted(set(d["alpha"] for d in group)):
                    ss = sorted([d for d in group if d["alpha"] == a], key=lambda d: d["r"])
                    axs[0, j].plot([d["r"] for d in ss], [d[field] for d in ss], color=col, ls=ls, alpha=0.17, lw=0.6)
                yy = [np.median([d[field] for d in group if d["r"] == r]) for r in rr]
                axs[0, j].plot(
                    rr,
                    yy,
                    ls,
                    color=col,
                    marker=mk,
                    ms=3,
                    lw=1.4,
                    label=f"{label}, " + ("HK" if dyn == "hk" else "Deffuant"),
                )
            for a in sorted(set(d["alpha"] for d in group)):
                ss = sorted([d for d in group if d["alpha"] == a], key=lambda d: d["r"])
                axs[1, j].plot(
                    [d["r"] for d in ss], [d["log_ratio"] for d in ss], color=COLOR[dyn], ls=ls, alpha=0.18, lw=0.7
                )
            yy = [np.median([d["log_ratio"] for d in group if d["r"] == r]) for r in rr]
            axs[1, j].plot(
                rr, yy, ls, color=COLOR[dyn], marker=mk, ms=3, lw=1.5, label="HK" if dyn == "hk" else "Deffuant"
            )
            for cross in crossings:
                if (
                    cross["source"] == "five-alpha median original grid"
                    and cross["dynamics"] == dyn
                    and cross["epsilon"] == eps
                    and cross["progress"] == 0.5
                ):
                    axs[1, j].scatter(
                        cross["roots"],
                        np.zeros(len(cross["roots"])),
                        marker="D",
                        s=30,
                        facecolors="white",
                        edgecolors=COLOR[dyn],
                        zorder=5,
                    )
            ss = sorted([d for d in extra if d["dynamics"] == dyn and d["epsilon"] == eps], key=lambda d: d["r"])
            for field, col in [("TE50", "#177e89"), ("TN50", "#d17b0f")]:
                axs[0, j].scatter(
                    [d["r"] for d in ss],
                    [d[field] for d in ss],
                    s=12,
                    facecolors="white",
                    edgecolors=col,
                    lw=0.65,
                    zorder=4,
                )
            axs[1, j].scatter(
                [d["r"] for d in ss],
                [d["log_ratio"] for d in ss],
                s=12,
                facecolors="white",
                edgecolors=COLOR[dyn],
                lw=0.65,
                zorder=4,
            )
        for dyn in ("hk", "deffuant"):
            g = [d for d in rows if d["dynamics"] == dyn and d["epsilon"] == eps and d["r"] <= 100]
            bad = sorted(set(d["r"] for d in g if not d["converged"]))
            for field in ("TE50", "TN50"):
                axs[0, j].scatter(
                    bad,
                    [np.median([d[field] for d in g if d["r"] == r]) for r in bad],
                    marker="x",
                    s=22,
                    c="black",
                    lw=0.65,
                    zorder=6,
                )
            axs[1, j].scatter(
                bad,
                [np.median([d["log_ratio"] for d in g if d["r"] == r]) for r in bad],
                marker="x",
                s=22,
                c="black",
                lw=0.65,
                zorder=6,
            )
        axs[0, j].set_yscale("log")
        axs[1, j].axhline(0, color="#777777", lw=0.8)
        for ax in axs[:, j]:
            ax.set_xscale("log")
            ax.grid(alpha=0.16)
            ax.set_xlim(0.085, 115)
        axs[1, j].set_xlabel(r"Relative rate $r=q/\alpha$")
    axs[0, 0].set_ylabel(r"Half-release time $T_{50}$ in $\tau=\alpha t$")
    axs[1, 0].set_ylabel(r"$\Delta_{50}=\log(T^E_{50}/T^\rho_{50})$")
    axs[0, 0].legend(frameon=False, fontsize=8)
    axs[1, 2].legend(frameon=False, fontsize=9)
    axs[1, 0].text(
        0.96, 0.96, "Edge slower", transform=axs[1, 0].transAxes, ha="right", va="top", fontsize=9, color="#555555"
    )
    axs[1, 0].text(0.03, 0.05, "Edge faster", transform=axs[1, 0].transAxes, fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.text(
        0.5,
        0.012,
        "Hollow circles: new runs at alpha = 0.01. Black crosses: at least one run has an unresolved terminal plateau.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.savefig(FIGURES / "edge_node_clocks.png")
    fig.savefig(FIGURES / "edge_node_clocks.pdf")
    plt.close(fig)
    # Compact paper-ready row, using exactly the same observable as the original top row.
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=False)
    for j, eps in enumerate((0.2, 0.4, 0.8)):
        ax = axs[j]
        for dyn in ("hk", "deffuant"):
            group = [d for d in rows if d["dynamics"] == dyn and d["epsilon"] == eps and d["r"] <= 100]
            rr = sorted(set(d["r"] for d in group))
            yy = [np.median([d["log_ratio"] for d in group if d["r"] == r]) for r in rr]
            for a in sorted(set(d["alpha"] for d in group)):
                ss = sorted([d for d in group if d["alpha"] == a], key=lambda d: d["r"])
                ax.plot([d["r"] for d in ss], [d["log_ratio"] for d in ss], color=COLOR[dyn], alpha=0.17, lw=0.7)
            ax.plot(
                rr,
                yy,
                "o-" if dyn == "hk" else "^--",
                color=COLOR[dyn],
                ms=3,
                lw=1.5,
                label="HK" if dyn == "hk" else "Deffuant",
            )
            ss = [d for d in extra if d["dynamics"] == dyn and d["epsilon"] == eps]
            ax.scatter(
                [d["r"] for d in ss],
                [d["log_ratio"] for d in ss],
                s=14,
                facecolors="white",
                edgecolors=COLOR[dyn],
                lw=0.7,
                zorder=4,
            )
            bad = sorted(set(d["r"] for d in group if not d["converged"]))
            ax.scatter(
                bad,
                [np.median([d["log_ratio"] for d in group if d["r"] == r]) for r in bad],
                marker="x",
                s=22,
                c="black",
                lw=0.65,
                zorder=6,
            )
            for cross in crossings:
                if (
                    cross["source"] == "five-alpha median original grid"
                    and cross["dynamics"] == dyn
                    and cross["epsilon"] == eps
                    and cross["progress"] == 0.5
                ):
                    ax.scatter(
                        cross["roots"],
                        np.zeros(len(cross["roots"])),
                        marker="D",
                        s=32,
                        facecolors="white",
                        edgecolors=COLOR[dyn],
                        zorder=5,
                    )
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(r"$\epsilon=" + str(eps) + "$")
        ax.set_xlabel(r"$r=q/\alpha$")
        ax.grid(alpha=0.16)
    axs[0].set_ylabel(r"$\log(T^E_{50}/T^\rho_{50})$")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "relative_clock.png")
    fig.savefig(FIGURES / "relative_clock.pdf")
    plt.close(fig)
    # Two-parameter summaries of clock curves are illustrative, not full-curve replacements.
    simple = []
    for dyn in ("hk", "deffuant"):
        for eps in (0.2, 0.4, 0.8):
            ds = [d for d in rows if d["dynamics"] == dyn and d["epsilon"] == eps and d["converged"] and d["r"] <= 100]
            x = np.log([d["r"] for d in ds])
            y = np.array([d["log_ratio"] for d in ds])
            if eps < 0.8:
                p, pred = fit_clock_ratio(x, y)
                label = "logratio = a - b log(r)"
            else:
                p, pred = fit_clock_ratio(x, y, softplus=True)
                label = "logratio = -log(1+(r/rc)^nu)"
            simple.append(
                dict(
                    dynamics=dyn,
                    epsilon=eps,
                    model=label,
                    parameters=list(p),
                    rms_logratio_error=float(np.sqrt(np.mean((pred - y) ** 2))),
                    max_logratio_error=float(max(abs(pred - y))),
                )
            )
    (DATA / "simple_clock_fits.json").write_text(json.dumps(simple, indent=2))
    print(json.dumps(crossings, indent=2))


if __name__ == "__main__":
    main()
