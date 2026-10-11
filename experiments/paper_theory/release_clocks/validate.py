"""Quantile continuity, out-of-rate prediction, and publication plots."""

import csv
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np

from theory.release.curves import norm, qpredict, quantiles
from theory.release.fitting import errors

from .analyze.config import LEVELS
from .analyze.curves import cases, key
from .analyze.tables import summarize, write_csv
from .artifact_paths import DATA, FIGURES, INPUT_RAW, SOURCE, ensure_output_dirs

LV = LEVELS["quantile7_mid"]
COLORS = {"hk": "#2166ac", "deffuant": "#c42935"}


from theory.release.curves import interpolate_times


def main():
    """Validate archived-rate anchors and independently refined crossing/delayed-cascade curves; write error evidence."""
    ensure_output_dirs()
    cs = list(cases())
    cmap = {key(c): c for c in cs}
    desc = []
    qs = {}
    base = {}
    with (SOURCE / "edge_timescale_fits.csv").open() as f:
        for row in csv.DictReader(f):
            base[
                (
                    row["dynamics"],
                    round(float(row["alpha"]), 12),
                    float(row["epsilon"]),
                    round(float(row["q_over_alpha"]), 9),
                )
            ] = row
    for c in cs:
        qt = quantiles(c["t"], c["se"], LV)
        qs[key(c)] = qt
        vals = quantiles(c["t"], c["se"], [0.05, 0.1, 0.25, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99])
        nt = quantiles(c["t"], c["sn"], [0.5])[0]
        d = {k: c[k] for k in ("dynamics", "epsilon", "alpha", "r", "converged")}
        d.update(
            {f"T{int(p * 100):02}": float(v) for p, v in zip([0.05, 0.1, 0.25, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99], vals)}
        )
        d["delta50"] = float(np.log(d["T50"] / nt))
        d["duration90"] = d["T95"] - d["T05"]
        d["asymmetry"] = float(np.log((d["T95"] - d["T50"]) / (d["T50"] - d["T05"])))
        d["node_T50"] = nt
        desc.append(d)
    write_csv(DATA / "recommended_parameters.csv", desc)
    cv = []
    for dyn in ("hk", "deffuant"):
        for eps in (0.2, 0.4, 0.8):
            for a in sorted(set(c["alpha"] for c in cs)):
                group = sorted(
                    [c for c in cs if c["dynamics"] == dyn and c["epsilon"] == eps and c["alpha"] == a and c["r"] > 0],
                    key=lambda c: c["r"],
                )
                rr = np.array([c["r"] for c in group])
                qq = np.array([qs[key(c)] for c in group])
                for i, c in enumerate(group[1:-1], 1):
                    keep = np.arange(len(group)) != i
                    qt = interpolate_times(rr[keep], qq[keep], rr[i])
                    pred = qpredict(c["t"], LV, qt)
                    cv.append(
                        dict(
                            dynamics=dyn,
                            epsilon=eps,
                            alpha=a,
                            r=c["r"],
                            converged=c["converged"],
                            model="quantile7_mid_rate_holdout",
                            **errors(c["t"], c["se"], pred),
                        )
                    )
    write_csv(DATA / "rate_holdout.csv", cv)
    summary = {
        "rate_holdout": summarize(cv),
        "monotone_correction_max": float(max(np.max(c["se"] - np.minimum.accumulate(c["se"])) for c in cs)),
        "source_files": {
            "raw_shards/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (SOURCE / "raw_shards").glob("*.npz")
        },
    }
    # Recompute all quantile errors under shorter normalization tails (observed converged curves only).
    sens = []
    for c in cs:
        if not c["converged"] or c["r"] == 0:
            continue
        energy = c["a"] * c["se"] + c["c"]
        baseqt = qs[key(c)]
        for count in (51, 201):
            plateau = np.median(energy[-min(count, len(energy)) :])
            sy = (energy - plateau) / (energy[0] - plateau)
            qt = quantiles(c["t"], sy, LV)
            sens.append(
                dict(
                    dynamics=c["dynamics"],
                    epsilon=c["epsilon"],
                    alpha=c["alpha"],
                    r=c["r"],
                    tail_count=count,
                    max_relative_T_change=float(max(abs(qt / baseqt - 1))),
                )
            )
    write_csv(DATA / "plateau_sensitivity.csv", sens)
    summary["plateau_sensitivity"] = {
        str(n): {
            k: float(v)
            for k, v in zip(
                ("median", "p90", "max"),
                np.quantile([r["max_relative_T_change"] for r in sens if r["tail_count"] == n], [0.5, 0.9, 1]),
            )
        }
        for n in (51, 201)
    }
    # New simulations are shape predictions from the original lattice; their plateaus are observed.
    refined = []
    anchors = []
    newdesc = []
    for p in sorted((INPUT_RAW).glob("*.npz")):
        z = np.load(p)
        dyn = str(z["dynamics"])
        eps = float(z["epsilon"])
        a = float(z["alpha"])
        r = float(z["r"])
        t = z["tau"]
        se, _, _ = norm(z["edge_energy"])
        sn, _, _ = norm(z["node_energy"])
        group = sorted(
            [c for c in cs if c["dynamics"] == dyn and c["epsilon"] == eps and c["alpha"] == a and c["r"] > 0],
            key=lambda c: c["r"],
        )
        qt = interpolate_times(np.array([c["r"] for c in group]), np.array([qs[key(c)] for c in group]), r)
        refined.append(
            dict(
                dynamics=dyn,
                epsilon=eps,
                alpha=a,
                r=r,
                converged=bool(z["converged"]),
                model="new_rate_prediction",
                **errors(t, se, qpredict(t, LV, qt)),
            )
        )
        actual = quantiles(t, se, LV)
        actual_node = quantiles(t, sn, [0.5])[0]
        newdesc.append(
            dict(
                dynamics=dyn,
                epsilon=eps,
                r=r,
                T50=actual[2],
                delta50=float(np.log(actual[2] / actual_node)),
                duration90=actual[5] - actual[0],
                asymmetry=float(np.log((actual[5] - actual[2]) / (actual[2] - actual[0]))),
            )
        )
        if r == 1.5625:
            old = cmap[(dyn, round(a, 12), eps, round(r, 9))]
            anchors.append(
                dict(
                    dynamics=dyn,
                    epsilon=eps,
                    max_energy_difference=float(
                        np.max(abs(z["edge_energy"] - np.interp(t, old["t"], old["se"] * old["a"] + old["c"])))
                    ),
                )
            )
    if refined:
        write_csv(DATA / "new_rate_validation.csv", refined)
        write_csv(DATA / "new_rate_parameters.csv", newdesc)
        summary["new_rate_all"] = summarize(refined)
        summary["new_rate_crossover"] = summarize([r for r in refined if r["r"] != 1.5625 and r["r"] < 3])
        summary["new_rate_delayed_window"] = summarize([r for r in refined if r["r"] > 3])
        summary["solver_anchors"] = anchors
    (DATA / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    # Main replacement parameter figure: quantile clocks, no hidden branch selection.
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    fig, axs = plt.subplots(3, 3, figsize=(11, 9), sharex=True)
    for j, eps in enumerate((0.2, 0.4, 0.8)):
        axs[0, j].set_title(r"$\epsilon=" + str(eps) + "$")
        for dyn in ("hk", "deffuant"):
            group = [d for d in desc if d["dynamics"] == dyn and d["epsilon"] == eps and 0 < d["r"] <= 100]
            for a in sorted(set(d["alpha"] for d in group)):
                sub = sorted([d for d in group if d["alpha"] == a], key=lambda d: d["r"])
                for i, name in enumerate(("delta50", "duration90", "asymmetry")):
                    axs[i, j].plot([d["r"] for d in sub], [d[name] for d in sub], color=COLORS[dyn], alpha=0.16, lw=0.7)
            rr = sorted(set(d["r"] for d in group))
            for i, name in enumerate(("delta50", "duration90", "asymmetry")):
                yy = [np.median([d[name] for d in group if d["r"] == r]) for r in rr]
                axs[i, j].plot(
                    rr,
                    yy,
                    "o-" if dyn == "hk" else "^--",
                    color=COLORS[dyn],
                    ms=3,
                    lw=1.3,
                    label="HK" if dyn == "hk" else "Deffuant",
                )
            if j < 2:
                sub = [d for d in newdesc if d["dynamics"] == dyn and d["epsilon"] == eps and d["r"] < 3]
                for i, name in enumerate(("delta50", "duration90", "asymmetry")):
                    axs[i, j].scatter(
                        [d["r"] for d in sub],
                        [d[name] for d in sub],
                        s=16,
                        facecolors="none",
                        edgecolors=COLORS[dyn],
                        lw=0.7,
                        zorder=4,
                    )
        for i in range(3):
            axs[i, j].set_xscale("log")
            axs[i, j].grid(alpha=0.16)
        axs[0, j].axhline(0, color="gray", lw=0.7)
        axs[1, j].set_yscale("log")
        axs[2, j].set_xlabel(r"$r=q/\alpha$")
    axs[0, 0].set_ylabel(r"$\log(T^E_{50}/T^\rho_{50})$")
    axs[1, 0].set_ylabel(r"Release duration $T_{95}-T_{05}$")
    axs[2, 0].set_ylabel(r"$\log[(T_{95}-T_{50})/(T_{50}-T_{05})]$")
    axs[0, 0].legend(frameon=False)
    fig.suptitle("Observable energy-release clocks across the crossover", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURES / "continuous_parameters.png")
    fig.savefig(FIGURES / "continuous_parameters.pdf")
    plt.close(fig)
    # Curve examples chosen to include both crossings, separated clocks, and delayed cascade.
    specs = [
        ("hk", 0.2, 1.5625),
        ("hk", 0.4, 2.209708691207961),
        ("hk", 0.8, 100.0),
        ("hk", 0.2, 100.0),
        ("deffuant", 0.2, 17.67766952966369),
        ("deffuant", 0.2, 25.0),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (dyn, eps, r) in zip(axs.flat, specs):
        c = min(
            [c for c in cs if c["dynamics"] == dyn and c["epsilon"] == eps and c["alpha"] == 0.01],
            key=lambda c: abs(c["r"] - r),
        )
        t = c["t"]
        y = c["se"]
        b = base[key(c)]
        w, k, beta = [float(b[n]) for n in ("weight", "kappa", "beta")]
        pk = w * np.exp(-((k * c["r"] * t) ** beta)) + (1 - w) * c["sn"]
        pq = qpredict(t, LV, qs[key(c)])
        mask = t > 0
        ax.plot(t[mask], y[mask], color="black", lw=1.8, label="Solver")
        ax.plot(t[mask], pk[mask], color="#7b3294", ls="--", lw=1.3, label="KWW + node")
        ax.plot(t[mask], pq[mask], color="#008c78", lw=1.2, label="7 release times")
        ax.scatter(qs[key(c)], 1 - LV, s=18, color="#008c78", zorder=4)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.15)
        ax.set_title(f"{dyn.upper()}, " + r"$\epsilon=" + str(eps) + r",\ r=" + f"{c['r']:.3g}" + r"$")
        ax.text(
            0.04,
            0.08,
            f"RMSE: {errors(t, y, pk)['rmse']:.4f} / {errors(t, y, pq)['rmse']:.4f}",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_xlabel(r"$\tau=\alpha t$")
        ax.set_ylabel(r"$S_E$")
    axs[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "curve_comparison.png")
    fig.savefig(FIGURES / "curve_comparison.pdf")
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "source_files"}, indent=2))


if __name__ == "__main__":
    main()
