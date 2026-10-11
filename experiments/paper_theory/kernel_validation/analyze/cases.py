"""Exact-versus-sampled finite-list moments and increment-law discrepancies."""

from __future__ import annotations

import math

import numpy as np

from common.io.codec import decoded
from common.numerics.distributions import hellinger
from theory.kernels import deposit, exact_row


def analyze(record):
    """Return scalar and distribution evidence comparing one frozen closure with finite-list moments or sampled increments."""
    job = record["job"]
    meta = job["meta"]
    f = record["result"]
    axis = np.asarray(f["axis"])
    b = len(axis)
    if meta["scan"] == "mixed_term":
        rho = decoded(job["payload"]["request"]["initial"]["probabilities"])
        mu = np.asarray(f["mean"])
        coef = meta["alpha"] ** 2 * meta["physical_dt"] * np.outer(mu, mu)
        transition = np.asarray(f["transition"]).reshape(b, b)
        deposited_increment = transition @ axis - axis
        deposited_defect = np.outer(deposited_increment, deposited_increment) / meta["physical_dt"]
        return [
            {
                **meta,
                "id": job["id"],
                "mixed_coefficient_mean_abs": float(np.sum(np.outer(rho, rho) * np.abs(coef))),
                "xy_weak_defect": float(rho @ coef @ rho),
                "deposited_xy_weak_defect": float(rho @ deposited_defect @ rho),
                "deposited_mixed_mean_abs": float(np.sum(np.outer(rho, rho) * np.abs(deposited_defect))),
                "deposition_mixed_bias_l1": float(np.sum(np.outer(rho, rho) * np.abs(deposited_defect - coef))),
                "missing_term_order": "alpha^2 physical_dt",
                "source": None,
            }
        ], []
    rows = []
    curves = []
    rng = np.random.default_rng(meta["sample_seed"])
    normal = np.asarray(f["transition"]).reshape(b, b)
    for source in sorted({int(np.argmin(abs(axis - x))) for x in meta["source_locations"]}):
        exact, mean, second, empty, tail = exact_row(f, meta, source)
        n, m = meta["degree"], meta["recommendations"]
        count = meta["samples"]
        p = np.asarray(f["neighbor_kernel"]).reshape(b, b)[source]
        q = np.asarray(f["recommendation_kernel"]).reshape(b, b)[source]
        targets = np.column_stack((rng.choice(b, (count, n), p=p), rng.choice(b, (count, m), p=q)))
        diff = axis[targets] - axis[source]
        ok = abs(diff) <= meta["epsilon"] + 1e-14
        counts = ok.sum(axis=1)
        hk = np.divide((diff * ok).sum(axis=1), counts, out=np.zeros(count), where=counts > 0)
        dh = np.divide((diff * diff * ok).sum(axis=1), counts, out=np.zeros(count), where=counts > 0)
        within = dh - hk * hk
        sampled = deposit(axis, axis[source] + meta["alpha"] * meta["h"] * hk, np.full(count, 1 / count))
        row = {
            **meta,
            "id": job["id"],
            "source": float(axis[source]),
            "normal_exact_hellinger": hellinger(normal[source], exact),
            "normal_exact_tv": float(0.5 * abs(normal[source] - exact).sum()),
            "mc_exact_tv": float(0.5 * abs(sampled - exact).sum()),
            "exact_mean": mean,
            "solver_mean": f["mean"][source],
            "exact_hk_second": second,
            "solver_hk_second": f["hk_second"][source],
            "mean_error": abs(mean - f["mean"][source]),
            "second_error": abs(second - f["hk_second"][source]),
            "empty_mass": empty,
            "normal_unbounded_tail": tail,
            "same_list_variance": float(within.mean()),
            "same_list_variance_se": float(within.std(ddof=1) / math.sqrt(count)),
            "moment_gap": f["deffuant_second"][source] - f["hk_second"][source],
            "mc_deffuant_second": float(dh.mean()),
            "mc_deffuant_second_se": float(dh.std(ddof=1) / math.sqrt(count)),
            "mc_empty_mass": float(np.mean(counts == 0)),
            "gap_minus_list_variance": float(f["deffuant_second"][source] - f["hk_second"][source] - within.mean()),
            "mc_hk_mean": float(hk.mean()),
            "mc_hk_mean_se": float(hk.std(ddof=1) / math.sqrt(count)),
            "deposited_mean_bias": float(
                np.dot(normal[source], axis - axis[source]) - meta["alpha"] * meta["h"] * mean
            ),
        }
        rows.append(row)
        curves.append(
            {
                "id": job["id"],
                "source": float(axis[source]),
                "axis": axis.tolist(),
                "exact": exact.tolist(),
                "normal": normal[source].tolist(),
                "mc": sampled.tolist(),
            }
        )
    return rows, curves
