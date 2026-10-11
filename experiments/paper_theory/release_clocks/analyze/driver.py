"""Offline comparison of quantile interpolants and empirical survival models."""

import argparse
import csv
import json

import numpy as np

from experiments.paper_theory.release_clocks.analyze import config
from experiments.paper_theory.release_clocks.artifact_paths import ensure_output_dirs


def run(args):
    """Read archived curves/fit parameters and write model metrics plus quantile descriptors for the selected pilot/full dataset."""
    from experiments.paper_theory.release_clocks.analyze.curves import cases, key
    from experiments.paper_theory.release_clocks.analyze.tables import summarize, write_csv
    from theory.release.curves import qpredict, quantiles
    from theory.release.fitting import errors, fit_model

    ensure_output_dirs()
    baseline = {}
    with (config.SOURCE / "edge_timescale_fits.csv").open() as f:
        for r in csv.DictReader(f):
            baseline[
                r["dynamics"], round(float(r["alpha"]), 12), float(r["epsilon"]), round(float(r["q_over_alpha"]), 9)
            ] = r
    rows = []
    descriptors = []
    for ic, c in enumerate(cases(args.pilot)):
        meta = {k: c[k] for k in ("dynamics", "alpha", "epsilon", "r", "converged")}
        t = c["t"]
        y = c["se"]
        old = baseline[key(c)]
        w, k, b = [float(old[n]) for n in ("weight", "kappa", "beta")]
        pred = w * np.exp(-((k * c["r"] * t) ** b)) + (1 - w) * c["sn"] if c["r"] else c["sn"]
        rows.append(dict(**meta, model="kww_node3", **errors(t, y, pred)))
        desc = dict(
            **meta,
            plateau=c["c"],
            amplitude=c["a"],
            monotone_correction=float(np.max(abs(np.minimum.accumulate(y) - y))),
        )
        for name, lv in config.LEVELS.items():
            qt = quantiles(t, y, lv)
            pred = qpredict(t, lv, qt)
            rows.append(dict(**meta, model=name, **errors(t, y, pred)))
            qt_even = quantiles(t[::2], y[::2], lv)
            rows.append(
                dict(**meta, model=name + "_time_holdout", **errors(t[1::2], y[1::2], qpredict(t[1::2], lv, qt_even)))
            )
        for lv, qt in zip(config.LEVELS["quantile9"], quantiles(t, y, config.LEVELS["quantile9"])):
            desc[f"T{int(lv * 100):02}"] = qt
        for lv, qt in zip([0.1, 0.5, 0.9], quantiles(t, c["sn"], [0.1, 0.5, 0.9])):
            desc[f"node_T{int(lv * 100):02}"] = qt
        descriptors.append(desc)
        if not args.quantiles_only:
            for name in ("weibull2", "gamma2", "gompertz2", "generalized_gamma3", "hazard3"):
                pred, info = fit_model(c, name)
                rows.append(dict(**meta, model=name, **errors(t, y, pred), **info))
        if (ic + 1) % 12 == 0:
            print(f"{ic + 1} curves", flush=True)
    suffix = "_pilot" if args.pilot else ""
    output_dir = config.PILOT if args.pilot else config.DATA
    write_csv(output_dir / f"metrics{suffix}.csv", rows)
    write_csv(output_dir / f"descriptors{suffix}.csv", descriptors)
    (output_dir / f"summary{suffix}.json").write_text(json.dumps(summarize(rows), indent=2))
    print(json.dumps(summarize(rows)["converged_positive"], indent=2))


def main():
    """Parse pilot/quantiles-only options and dispatch offline survival-model analysis."""
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--quantiles-only", action="store_true")
    run(p.parse_args())
