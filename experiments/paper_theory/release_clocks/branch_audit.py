"""Check whether the beta<=5 bound alone causes crossover branch changes."""

import numpy as np

from theory.release.curves import norm
from theory.release.mixture import alternative_branch

from .analyze.tables import write_csv
from .artifact_paths import DATA, INPUT_RAW, ensure_output_dirs


def main():
    """Analyze alternative fast-layer fit branches and their reconstruction/identifiability evidence."""
    ensure_output_dirs()
    rows = []
    for path in sorted((INPUT_RAW).glob("*.npz")):
        z = np.load(path)
        r = float(z["r"])
        if r > 3:
            continue
        t = z["tau"]
        node, _, _ = norm(z["node_energy"])
        edge, _, _ = norm(z["edge_energy"])
        idx = np.unique(np.linspace(0, len(t) - 1, min(1501, len(t))).astype(int))
        tt = t[idx]
        sn = node[idx]
        se = edge[idx]
        for limit in (5.0, 50.0):
            fit = alternative_branch(tt, sn, se, r, limit)
            rows.append(
                dict(
                    dynamics=str(z["dynamics"]),
                    epsilon=float(z["epsilon"]),
                    r=r,
                    beta_upper=limit,
                    weight=float(fit.x[0]),
                    kappa=float(np.exp(fit.x[1])),
                    beta=float(np.exp(fit.x[2])),
                    fit_rmse=float(np.sqrt(np.mean(fit.fun**2))),
                    success=bool(fit.success),
                )
            )
        print(path.name, flush=True)
    write_csv(DATA / "branch_audit.csv", rows)


if __name__ == "__main__":
    main()
