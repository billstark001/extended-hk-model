"""Figures comparing matched population errors across independently initialized groups."""

from collections import defaultdict

import numpy as np


def plot(directory):
    """Render saved population summary tables into the study figure directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from common.plotting.figures import save

    from ...data import read_csv

    rows = read_csv(directory / "tables" / "population_summary.csv")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dynamics"], row["recommender"], row["r"])].append(row)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for key, members in groups.items():
        members.sort(key=lambda r: int(r["n"]))
        for axis, metric in zip(axes, ("micro_variance", "micro_discordant", "final_node_w1")):
            x = [int(r["n"]) for r in members]
            y = [float(r[f"{metric}_mean"]) for r in members]
            sem = np.array([float(r[f"{metric}_sem"]) for r in members])
            axis.errorbar(
                x, y, yerr=np.nan_to_num(sem, nan=0), marker=".", linewidth=1, label=f"{key[0]} / {key[1]} / r={key[2]}"
            )
            axis.set_xlabel("N")
            axis.set_ylabel(metric.replace("_", " "))
            axis.grid(alpha=0.15)
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, directory / "figures" / "finite_population")
    (directory / "figures" / "finite_population.caption.txt").write_text(
        "Matched uniform initial law, fixed k and R; independent initial graph/opinion draws with conditional "
        "process branches. Error bars are SEM across initializations after averaging branches; a single "
        "initialization has unavailable SEM. Variance is a morphology indicator, discordant-edge fraction an "
        "isolation indicator; neither is relabeled as the original paper's P or H. W1 compares binned final "
        "opinion measures. StructureRandom uses finite graph scores while structure_random_l0 uses a pair "
        "overlap proxy. This experiment tests finite-N robustness; it does not establish a large-N theorem.\n"
    )
