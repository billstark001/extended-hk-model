"""Figures from saved parameter-clock tables with fit-quality/convergence markers."""

import matplotlib.pyplot as plt
import numpy as np

from experiments.paper_theory.release_clocks.parameter_clocks import config


def group(rows, dyn, eps):
    """Select one dynamics/confidence group restricted to rate ratio <= 100."""
    return [r for r in rows if r["dynamics"] == dyn and r["epsilon"] == eps and (r["r"] <= 100)]


def med(sub, field):
    """Return sorted distinct rate ratios and within-ratio medians for field."""
    rr = np.array(sorted(set((d["r"] for d in sub))))
    yy = np.array([np.median([d[field] for d in sub if d["r"] == r]) for r in rr])
    return (rr, yy)


def decorate(ax, eps=None):
    """Apply the fixed logarithmic rate axis, grid and optional confidence title."""
    ax.set_xscale("log")
    ax.set_xlim(0.085, 115)
    ax.grid(alpha=0.17)
    if eps is not None:
        ax.set_title("$\\epsilon=" + str(eps) + "$")


def flags(ax, sub, field, weak=False, poor=False, censor=True):
    """Overlay requested convergence, weak-fast-component and poor-node-fit markers on within-ratio medians."""
    for test, mk, color in [
        (lambda d: not d["converged"], "x", "black"),
        (lambda d: d["fast_weak"], "x", "#777777"),
        (lambda d: d["node_poor"], "s", config.COL[sub[0]["dynamics"]]),
    ]:
        if (
            mk == "x"
            and color == "black"
            and (not censor)
            or (color == "#777777" and (not weak))
            or (mk == "s" and (not poor))
        ):
            continue
        bad = sorted(set((d["r"] for d in sub if test(d))))
        yy = [np.median([d[field] for d in sub if d["r"] == r]) for r in bad]
        if mk == "s":
            ax.scatter(bad, yy, s=26, facecolors="none", edgecolors=color, marker=mk, lw=0.8, zorder=6)
        else:
            ax.scatter(bad, yy, s=26, c=color, marker=mk, lw=0.8, zorder=6)


def save(fig, name, footer):
    """Save the parameter-clock PDF/PNG pair and its explanatory caption in the configured figure directory."""
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.text(0.5, 0.009, footer, ha="center", fontsize=8, color="#555555")
    fig.savefig(config.CLOCK_FIGURES / (name + ".png"))
    fig.savefig(config.CLOCK_FIGURES / (name + ".pdf"))
    plt.close(fig)


def plots(rows, qs):
    """Render fitted/observed clock, component-rate and tail comparisons from typed metric and quantile rows."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160})
    fig, axs = plt.subplots(3, 3, figsize=(11.8, 9), sharex=True)
    for j, eps in enumerate((0.2, 0.4, 0.8)):
        for dyn in config.COL:
            sub = group(rows, dyn, eps)
            style = "-" if dyn == "hk" else "--"
            mark = "o" if dyn == "hk" else "^"
            for fld, col, label in [("T_fast_63", "#177e89", "Rewiring component"), ("T_node_63", "#d17b0f", "Node")]:
                x, y = med(sub, fld)
                axs[0, j].plot(
                    x,
                    y,
                    style,
                    color=col,
                    marker=mark,
                    ms=3,
                    lw=1.3,
                    label=f"{label}, " + ("HK" if dyn == "hk" else "Deffuant"),
                )
            for i, fld in [(1, "log_component_clock"), (2, "log_mean_ratio")]:
                for a in sorted(set((r["alpha"] for r in sub))):
                    ss = sorted([r for r in sub if r["alpha"] == a], key=lambda d: d["r"])
                    axs[i, j].plot(
                        [d["r"] for d in ss], [d[fld] for d in ss], color=config.COL[dyn], alpha=0.15, lw=0.7
                    )
                x, y = med(sub, fld)
                axs[i, j].plot(
                    x,
                    y,
                    style,
                    color=config.COL[dyn],
                    marker=mark,
                    ms=3,
                    lw=1.4,
                    label="HK" if dyn == "hk" else "Deffuant",
                )
                flags(axs[i, j], sub, fld, weak=i == 1, poor=True)
            x, y = med(sub, "observed_log_mean_ratio")
            axs[2, j].plot(x, y, ":", color=config.COL[dyn], lw=1.6)
        for i in range(3):
            decorate(axs[i, j], eps if i == 0 else None)
        axs[0, j].set_yscale("log")
        axs[1, j].axhline(0, color="gray", lw=0.7)
        axs[2, j].axhline(0, color="gray", lw=0.7)
        axs[2, j].set_xlabel("$r=q/\\alpha$")
    axs[0, 0].set_ylabel("$T_{63.2\\%}$ of each fitted component")
    axs[1, 0].set_ylabel("$\\log(T_f^{*}/T_\\rho^{*})$")
    axs[2, 0].set_ylabel("$\\log(\\overline{T}_E/\\overline{T}_\\rho)$")
    axs[0, 0].legend(frameon=False, fontsize=7.5)
    axs[1, 2].legend(frameon=False, fontsize=9)
    save(
        fig,
        "fitted_parameter_clocks",
        "Dotted: observed finite-window mean ratio. Grey crosses: weak KWW shape. Squares: poor node fit. Black crosses: unconverged.",
    )
    fig, axs = plt.subplots(2, 3, figsize=(11.8, 6.7), sharex=True)
    for i, p in enumerate((0.9, 0.99)):
        for j, eps in enumerate((0.2, 0.4, 0.8)):
            ax = axs[i, j]
            for dyn in config.COL:
                sub = group([r for r in qs if r["p"] == p], dyn, eps)
                sty = "-" if dyn == "hk" else "--"
                mk = "o" if dyn == "hk" else "^"
                for field, col, label in [("edge_observed", "#177e89", "Edge"), ("node_observed", "#d17b0f", "Node")]:
                    for a in sorted(set((r["alpha"] for r in sub))):
                        ss = sorted([r for r in sub if r["alpha"] == a], key=lambda d: d["r"])
                        ax.plot([d["r"] for d in ss], [d[field] for d in ss], color=col, alpha=0.16, lw=0.7)
                    x, y = med(sub, field)
                    ax.plot(
                        x,
                        y,
                        sty,
                        color=col,
                        marker=mk,
                        ms=3,
                        lw=1.4,
                        label=f"{label}, " + ("HK" if dyn == "hk" else "Deffuant"),
                    )
                    flags(ax, sub, field)
            decorate(ax, eps if i == 0 else None)
            ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel("$T_{" + str(int(p * 100)) + "}$ in $\\tau=\\alpha t$")
            if i == 1:
                ax.set_xlabel("$r=q/\\alpha$")
    axs[0, 2].legend(frameon=False, fontsize=8)
    save(
        fig,
        "T90_T99_absolute",
        "Observed release times. Thick: five-alpha median; thin: individual alpha. Black crosses: at least one unconverged trajectory.",
    )
    fig, axs = plt.subplots(3, 3, figsize=(11.8, 9), sharex=True)
    for i, p in enumerate(config.P):
        for j, eps in enumerate((0.2, 0.4, 0.8)):
            ax = axs[i, j]
            for dyn in config.COL:
                sub = group([r for r in qs if r["p"] == p], dyn, eps)
                style = "-" if dyn == "hk" else "--"
                mk = "o" if dyn == "hk" else "^"
                for a in sorted(set((r["alpha"] for r in sub))):
                    ss = sorted([r for r in sub if r["alpha"] == a], key=lambda d: d["r"])
                    ax.plot(
                        [d["r"] for d in ss],
                        [d["observed_log_ratio"] for d in ss],
                        color=config.COL[dyn],
                        alpha=0.15,
                        lw=0.7,
                    )
                x, y = med(sub, "observed_log_ratio")
                ax.plot(
                    x,
                    y,
                    style,
                    color=config.COL[dyn],
                    marker=mk,
                    ms=3,
                    lw=1.5,
                    label="HK" if dyn == "hk" else "Deffuant",
                )
                x, y = med(sub, "full_model_log_ratio")
                ax.plot(x, y, ":", color=config.COL[dyn], lw=1.5)
                flags(ax, sub, "observed_log_ratio")
                flags(ax, sub, "full_model_log_ratio", censor=False, poor=True)
            decorate(ax, eps if i == 0 else None)
            ax.axhline(0, color="gray", lw=0.7)
            if j == 0:
                ax.set_ylabel("$\\log(T^E_{" + str(int(p * 100)) + "}/T^\\rho_{" + str(int(p * 100)) + "})$")
            if i == 2:
                ax.set_xlabel("$r=q/\\alpha$")
    axs[0, 2].legend(frameon=False, fontsize=9)
    save(
        fig,
        "T50_T90_T99_ratios",
        "Lines/markers: observed. Dotted: KWW mixture with fitted Gompertz node input. Squares: poor node fit. Black crosses: unconverged.",
    )
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8))
    colors = {0.5: "#3875ab", 0.9: "#be6636", 0.99: "#69953d"}
    for j, eps in enumerate((0.2, 0.4, 0.8)):
        ax = axs[j]
        for p in config.P:
            for dyn in config.COL:
                sub = group([r for r in qs if r["p"] == p], dyn, eps)
                x, y = med(sub, "observed_log_ratio")
                ax.plot(
                    x,
                    y,
                    "-" if dyn == "hk" else "--",
                    color=colors[p],
                    lw=1.5,
                    marker="o" if dyn == "hk" else "^",
                    ms=2.5,
                    label=f"{int(p * 100)}%, " + ("HK" if dyn == "hk" else "Deffuant"),
                )
                flags(ax, sub, "observed_log_ratio")
        decorate(ax, eps)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xlabel("$r=q/\\alpha$")
    axs[0].set_ylabel("$\\log(T^E_p/T^\\rho_p)$")
    axs[2].legend(frameon=False, fontsize=8, ncol=2)
    save(
        fig,
        "observed_progress_clock_ratios",
        "Half-release and late release can have different ordering. Black crosses: at least one unresolved terminal plateau.",
    )
