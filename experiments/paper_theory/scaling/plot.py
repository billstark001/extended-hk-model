"""Energy and normalized-release panels from new or archived measure-sweep curves."""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common.plotting.figures import save
from experiments.paper_theory.data import raw_release_cases
from experiments.paper_theory.runtime.measure_analysis import cached_cases
from theory.release import release_clock


def plot_runs(root, suite, profile, scan=None):
    """Render grouped trajectories in tau units; existing_scaling reads archived inputs, while other suites read their completed NPZ caches."""
    directory = root / suite / profile if suite != "existing_scaling" else root / "existing_scaling"
    cases = (
        list(cached_cases(directory))
        if suite != "existing_scaling"
        else [
            c
            for c in raw_release_cases(scan)
            if c["epsilon"] == 0.4 and any(np.isclose(c["r"], r, atol=1e-9, rtol=0) for r in (2.209708691, 12.5, 50.0))
        ]
    )
    if not cases:
        raise FileNotFoundError(f"no curves in {directory}")
    groups = defaultdict(list)
    for case in cases:
        groups[(case["dynamics"], case["recommender"], case["epsilon"], case["r"])].append(case)
    for (rule, rec, epsilon, ratio), group in groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
        for case in sorted(group, key=lambda c: (c["alpha"], c.get("population", 500))):
            label = f"N={case['population']}" if suite == "population_closure" else f"α={case['alpha']:g}"
            for col, name in enumerate(("node_energy", "edge_energy")):
                axes[0, col].plot(case["tau"], case[name], linewidth=1, label=label)
                clock = release_clock(case["tau"], case[name], converged=case["converged"])
                axes[1, col].plot(case["tau"], clock.survival, linewidth=1)
                axes[0, col].set_title(name.replace("_", " "))
                axes[1, col].set_xlabel(r"$\tau=\alpha t$")
                axes[1, col].set_ylabel("Remaining energy / initial release")
                axes[0, col].set_ylabel("Energy")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{rule.upper()}, {rec}, ε={epsilon:g}, r={ratio:g}")
        fig.tight_layout()
        stem = f"{suite}_{rule}_{rec}_e{epsilon:g}_r{ratio:g}"
        save(fig, directory / "figures" / stem)
    (directory / "figures" / "caption.txt").write_text(
        "Upper panels compare absolute energies without separate endpoint normalization. Lower panels "
        "show tail-median-normalized finite-window release; consult clock status and horizon in the CSV. "
        "Small-alpha reference curves approximate, rather than establish, the continuum limit. "
        "population_closure varies only the nominal N of an N-independent pair closure and does not prove microscopic N convergence.\n"
    )
