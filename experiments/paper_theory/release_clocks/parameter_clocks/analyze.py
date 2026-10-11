"""Compare parameter-derived means/quantiles with observed finite-window release."""

import json

import numpy as np
from scipy.special import gamma

from experiments.paper_theory.release_clocks.analyze.curves import cases, key
from experiments.paper_theory.release_clocks.analyze.tables import write_csv
from experiments.paper_theory.release_clocks.parameter_clocks import config
from theory.release.curves import log_rate_crossings
from theory.release.fitting import errors
from theory.release.models import truncated_fast, truncated_node


def metadata(c):
    """Select dynamics, confidence, alpha, rate ratio and convergence fields for a clock result row."""
    return {k: c[k] for k in ("dynamics", "epsilon", "alpha", "r", "converged")}


def generate():
    """Compute and save parameter means, release-time tables and tail sensitivity from existing fits/curves; return (metric_rows, quantile_rows)."""
    from experiments.paper_theory.release_clocks.parameter_clocks.data import load_fits
    from theory.release.models import observed_q, scaled_e1, sn, times_full, times_node

    nodes, edges = load_fits()
    rows = []
    qs = []
    tests = []
    sens = []
    for i, c in enumerate(cases()):
        if c["r"] == 0:
            continue
        n = nodes[key(c)]
        e = edges[key(c)]
        h = float(n["initial_hazard"])
        g = float(n["acceleration"])
        w = float(e["weight"])
        k = float(e["kappa"])
        b = float(e["beta"])
        lam = k * c["r"]
        a = h / g
        mn = scaled_e1(a) / g
        mf = gamma(1 + 1 / b) / lam
        me = w * mf + (1 - w) * mn
        tn63 = float(times_node(1 - np.exp(-1), h, g))
        tf63 = 1 / lam
        t = c["t"]
        node_model = sn(t, h, g)
        fast = np.exp(-((lam * t) ** b))
        edge_full = w * fast + (1 - w) * node_model
        edge_cond = w * fast + (1 - w) * c["sn"]
        mn_obs = float(np.trapezoid(np.maximum(c["sn"], 0), t))
        me_obs = float(np.trapezoid(np.maximum(c["se"], 0), t))
        horizon = float(t[-1])
        mn_trunc = truncated_node(horizon, h, g)
        mf_trunc = truncated_fast(horizon, lam, b)
        me_cond = w * mf_trunc + (1 - w) * mn_obs
        weak = e["shape_identifiable"] != "True"
        poor = float(n["maximum_absolute_relative_error"]) > 0.2
        row = dict(
            **metadata(c),
            h0=h,
            g=g,
            w=w,
            kappa=k,
            beta=b,
            lambda_fast=lam,
            fast_weak=weak,
            node_poor=poor,
            horizon=horizon,
            T_fast_63=tf63,
            T_node_63=tn63,
            log_component_clock=float(np.log(tf63 / tn63)),
            mean_node=mn,
            mean_fast=mf,
            mean_edge=me,
            log_mean_ratio=float(np.log(me / mn)),
            observed_mean_node=mn_obs,
            observed_mean_edge=me_obs,
            observed_log_mean_ratio=float(np.log(me_obs / mn_obs)),
            conditional_log_mean_ratio=float(np.log(me_cond / mn_obs)),
            model_mean_node_tail_fraction=float(1 - mn_trunc / mn),
            model_mean_edge_tail_fraction=float(1 - (w * mf_trunc + (1 - w) * mn_trunc) / me),
            **{"full_" + k: v for k, v in errors(t, c["se"], edge_full).items()},
            **{"conditional_" + k: v for k, v in errors(t, c["se"], edge_cond).items()},
        )
        rows.append(row)
        for p in config.P:
            te = observed_q(t, c["se"], p)
            tn = observed_q(t, c["sn"], p)
            nf = float(times_node(p, h, g))
            ff = float((-np.log1p(-p)) ** (1 / b) / lam)
            ef = times_full(p, h, g, w, lam, b)
            ec = observed_q(t, edge_cond, p)
            qs.append(
                dict(
                    **metadata(c),
                    p=p,
                    edge_observed=te,
                    node_observed=tn,
                    edge_full_model=ef,
                    node_model=nf,
                    fast_component=ff,
                    edge_conditional_model=ec,
                    observed_log_ratio=float(np.log(te / tn)),
                    full_model_log_ratio=float(np.log(ef / nf)),
                    conditional_log_ratio=float(np.log(ec / tn)),
                    fast_log_ratio=float(np.log(ff / nf)),
                    node_relative_error=nf / tn - 1,
                    edge_relative_error=ef / te - 1,
                    conditional_edge_relative_error=ec / te - 1,
                    full_edge_beyond_horizon=ef > horizon,
                    full_node_beyond_horizon=nf > horizon,
                    fast_weak=weak,
                    node_poor=poor,
                )
            )
            if c["converged"] and p >= 0.9:
                for count in (51, 201):
                    vals = []
                    for sy in (c["sn"], c["se"]):
                        tail = float(np.median(sy[-min(count, len(sy)) :]))
                        vals.append(observed_q(t, (sy - tail) / (sy[0] - tail), p))
                    sens.append(
                        dict(
                            **metadata(c),
                            p=p,
                            tail_count=count,
                            node_relative_change=vals[0] / tn - 1,
                            edge_relative_change=vals[1] / te - 1,
                        )
                    )
        if i % 60 == 0:
            tn99 = float(times_node(1 - 1e-10, h, g))
            num = truncated_node(tn99, h, g)
            tests.append(
                dict(
                    case=list(key(c)),
                    node_integral_relative_error=abs(num / mn - 1),
                    mixture_mean_identity_error=abs(me / mn - (1 + w * (mf / mn - 1))),
                    node_limit_T90=abs(times_full(0.9, h, g, 0.0, lam, b) - float(times_node(0.9, h, g))),
                    fast_limit_T90=abs(times_full(0.9, h, g, 1.0, lam, b) - float((-np.log(0.1)) ** (1 / b) / lam)),
                )
            )
    write_csv(config.OUT / "parameter_metrics.csv", rows)
    write_csv(config.OUT / "release_times.csv", qs)
    write_csv(config.OUT / "tail_sensitivity.csv", sens)
    summary = {"formula_checks": tests, "quantile_errors": {}, "curve_errors": {}, "crossings": []}
    for name, condition in [
        ("converged_all", lambda r: r["converged"]),
        ("converged_common", lambda r: r["converged"] and r["r"] <= 100),
        ("common_good_node", lambda r: r["converged"] and r["r"] <= 100 and (not r["node_poor"])),
    ]:
        subset = [r for r in rows if condition(r)]
        summary["curve_errors"][name] = {}
        for m in ("full", "conditional"):
            v = [r[m + "_rmse"] for r in subset]
            summary["curve_errors"][name][m] = dict(
                n=len(v), median=float(np.median(v)), p90=float(np.quantile(v, 0.9)), max=float(max(v))
            )
        summary["quantile_errors"][name] = {}
        for p in config.P:
            group = [r for r in qs if condition(r) and r["p"] == p]
            stats = {}
            for k in ("node_relative_error", "edge_relative_error", "conditional_edge_relative_error"):
                vs = np.abs([r[k] for r in group])
                valid = np.isfinite(vs)
                v = vs[valid]
                stats[k] = dict(
                    n=int(sum(valid)),
                    unavailable=int(sum(~valid)),
                    median=float(np.median(v)),
                    p90=float(np.quantile(v, 0.9)),
                    max=float(max(v)),
                )
            summary["quantile_errors"][name][str(p)] = stats
    summary["mean_errors_common"] = {}
    for fld in ("log_mean_ratio", "conditional_log_mean_ratio"):
        d = np.array([r[fld] - r["observed_log_mean_ratio"] for r in rows if r["converged"] and r["r"] <= 100])
        summary["mean_errors_common"][fld] = dict(
            median_absolute=float(np.median(abs(d))),
            p90_absolute=float(np.quantile(abs(d), 0.9)),
            max_absolute=float(max(abs(d))),
        )
    summary["tail_normalization_max_relative_change"] = {
        str(p): {
            k: float(max((abs(r[k]) for r in sens if r["p"] == p)))
            for k in ("node_relative_change", "edge_relative_change")
        }
        for p in (0.9, 0.99)
    }
    for dyn in config.COL:
        for eps in (0.2, 0.4, 0.8):
            for p in config.P:
                sub = [
                    r for r in qs if r["dynamics"] == dyn and r["epsilon"] == eps and (r["p"] == p) and (r["r"] <= 100)
                ]
                rr = np.array(sorted(set((r["r"] for r in sub))))
                yy = np.array([np.median([r["observed_log_ratio"] for r in sub if r["r"] == x]) for x in rr])
                roots = log_rate_crossings(np.log(rr), yy)
                summary["crossings"].append(dict(dynamics=dyn, epsilon=eps, p=p, roots=roots))
    (config.OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    return (rows, qs)
