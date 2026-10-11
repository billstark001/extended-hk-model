"""CSV exports and grouped error summaries for survival-model comparisons."""

import csv

import numpy as np


def write_csv(path, rows):
    """Write heterogeneous rows using the union of table fields."""
    fields = list(dict.fromkeys((k for row in rows for k in row)))
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    """Aggregate error statistics by model and declared convergence/rate subsets."""
    out = {}
    for subset in ("converged_positive", "all_positive", "zero"):
        out[subset] = {}
        for name in sorted(set((r["model"] for r in rows))):
            rs = [
                r
                for r in rows
                if r["model"] == name
                and (
                    r["r"] > 0 and r["converged"]
                    if subset == "converged_positive"
                    else r["r"] > 0
                    if subset == "all_positive"
                    else r["r"] == 0
                )
            ]
            if not rs:
                continue
            v = np.array([r["rmse"] for r in rs])
            a = np.array([r["active_rmse"] for r in rs])
            out[subset][name] = dict(
                n=len(v),
                median=float(np.median(v)),
                p90=float(np.quantile(v, 0.9)),
                max=float(max(v)),
                above_003=int(sum(v > 0.03)),
                active_median=float(np.median(a)),
                active_p90=float(np.quantile(a, 0.9)),
                max_pointwise=float(max((r["max_error"] for r in rs))),
            )
    return out
