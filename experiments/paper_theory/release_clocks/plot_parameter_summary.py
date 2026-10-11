"""Compact companions using the saved parameter-clock calculation."""

import csv

import matplotlib.pyplot as plt

from .artifact_paths import CLOCK_INPUT_DATA, ensure_output_dirs
from .parameter_clocks.config import COL
from .parameter_clocks.plot import decorate, flags, group, med, save


def main():
    """Redraw compact parameter-mean and component-rate panels from saved clock tables."""
    ensure_output_dirs()
    rows = []
    for raw in csv.DictReader((CLOCK_INPUT_DATA / "parameter_metrics.csv").open()):
        row = {}
        for k, v in raw.items():
            if k == "dynamics":
                row[k] = v
            elif v in ("True", "False"):
                row[k] = v == "True"
            else:
                row[k] = float(v)
        rows.append(row)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 170})
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.7))
    for j, eps in enumerate((0.2, 0.4, 0.8)):
        ax = axs[j]
        for dyn in COL:
            sub = group(rows, dyn, eps)
            for a in sorted(set(r["alpha"] for r in sub)):
                ss = sorted([r for r in sub if r["alpha"] == a], key=lambda d: d["r"])
                ax.plot([d["r"] for d in ss], [d["log_mean_ratio"] for d in ss], color=COL[dyn], alpha=0.2, lw=0.7)
            x, y = med(sub, "log_mean_ratio")
            ax.plot(
                x,
                y,
                "o-" if dyn == "hk" else "^--",
                color=COL[dyn],
                ms=3,
                lw=1.5,
                label="HK" if dyn == "hk" else "Deffuant",
            )
            x, y = med(sub, "observed_log_mean_ratio")
            ax.plot(x, y, ":", color=COL[dyn], lw=1.6)
            flags(ax, sub, "log_mean_ratio", poor=True)
        decorate(ax, eps)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xlabel(r"$r=q/\alpha$")
    axs[0].set_ylabel(r"$\log(\overline{T}_E/\overline{T}_\rho)$")
    axs[2].legend(frameon=False, fontsize=9)
    save(
        fig,
        "mean_clock_from_parameters",
        "Lines/markers: original fit parameters. Dotted: observed finite-window area. Squares: poor node fit. Crosses: unconverged.",
    )
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.9))
    for j, eps in enumerate((0.2, 0.4, 0.8)):
        ax = axs[j]
        for dyn in COL:
            sub = group(rows, dyn, eps)
            for fld, col, label in [
                ("lambda_fast", "#177e89", r"$\kappa r$"),
                ("h0", "#d17b0f", r"$h_0$"),
                ("g", "#7851a9", r"$g$"),
            ]:
                x, y = med(sub, fld)
                ax.plot(
                    x,
                    y,
                    "-" if dyn == "hk" else "--",
                    color=col,
                    lw=1.4,
                    label=label + (", HK" if dyn == "hk" else ", Deffuant"),
                )
            flags(ax, sub, "lambda_fast", weak=True)
        decorate(ax, eps)
        ax.set_yscale("log")
        ax.set_xlabel(r"$r=q/\alpha$")
    axs[0].set_ylabel(r"Rate parameters (units of $\tau^{-1}$)")
    axs[1].legend(frameon=False, fontsize=8, ncol=2)
    save(
        fig,
        "original_rate_parameters",
        "These parameters have different meanings: KWW inverse scale, initial node hazard, and node hazard growth rate.",
    )


if __name__ == "__main__":
    main()
