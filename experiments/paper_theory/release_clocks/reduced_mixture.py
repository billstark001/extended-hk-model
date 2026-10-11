"""Two fitted parameters: original mixture with a fixed KWW exponent."""

import json

import numpy as np

from theory.release.fitting import errors
from theory.release.mixture import fixed_exponent

from .analyze.curves import cases
from .analyze.tables import summarize, write_csv
from .artifact_paths import DATA, ensure_output_dirs


def main():
    """Fit reduced two-parameter edge mixtures at fixed KWW exponents and summarize reconstruction errors."""
    ensure_output_dirs()
    rows = []
    for i, c in enumerate(cases()):
        if c["r"] == 0:
            continue
        idx = np.unique(np.linspace(0, len(c["t"]) - 1, min(1501, len(c["t"]))).astype(int))
        t = c["t"][idx]
        n = c["sn"][idx]
        y = c["se"][idx]
        for beta in (1.0, 1.5, 2.0):
            w, k, pred = fixed_exponent(c, t, n, y, beta)
            rows.append(
                dict(
                    **{k: c[k] for k in ("dynamics", "epsilon", "alpha", "r", "converged")},
                    model=f"fixed_beta_{beta:g}",
                    weight=w,
                    kappa=k,
                    beta=beta,
                    **errors(c["t"], c["se"], pred),
                )
            )
        if (i + 1) % 120 == 0:
            print(i + 1, flush=True)
    write_csv(DATA / "reduced_mixture.csv", rows)
    (DATA / "reduced_summary.json").write_text(json.dumps(summarize(rows), indent=2))
    print(json.dumps(summarize(rows)["converged_positive"], indent=2))


if __name__ == "__main__":
    main()
