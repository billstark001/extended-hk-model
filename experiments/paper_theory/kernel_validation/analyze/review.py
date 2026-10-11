"""Finite-list moment, Monte Carlo and mixed-increment evidence for manuscript summaries."""

from __future__ import annotations

import numpy as np

from common.io.codec import decoded
from experiments.paper_theory.reporting import raw, summarize, write_csv


def review_kernel(root, out, summary, kernel):
    """Summarize frozen-kernel exact moments, sampled-law gaps and mixed terms into tables and the supplied manuscript summary."""
    laws = [r for r in kernel if r["scan"] == "hk_law"]
    summary["kernel_groups"] = summarize(
        laws,
        ("degree", "quadrature"),
        ("normal_exact_tv", "normal_exact_hellinger", "mc_exact_tv", "normal_unbounded_tail"),
    )
    summary["kernel_max_mean_error"] = max((float(r["mean_error"]) for r in laws))
    summary["kernel_max_second_error"] = max((float(r["second_error"]) for r in laws))
    summary["kernel_max_gap_mc_se_units"] = max(
        (
            abs(float(r["gap_minus_list_variance"])) / float(r["same_list_variance_se"])
            for r in laws
            if float(r["same_list_variance_se"]) > 0
        )
    )
    mixed = []
    for row in kernel:
        if row["scan"] != "mixed_term":
            continue
        record = raw(root, "kernel", row["id"])
        rho = decoded(record["job"]["payload"]["request"]["initial"]["probabilities"])
        g = record["result"]["hk_second" if row["dynamics"] == "hk" else "deffuant_second"]
        diagonal = float(row["alpha"]) ** 2 * float(row["physical_dt"]) * float(rho @ np.asarray(g))
        mixed.append(
            {
                **row,
                "weighted_diagonal_sum": diagonal,
                "abs_mixed_to_diagonal_sum": float(row["mixed_coefficient_mean_abs"]) / diagonal,
            }
        )
    write_csv(out / "mixed_coefficient_review.csv", mixed)
    summary["mixed_coefficients"] = [
        {
            k: r[k]
            for k in (
                "dynamics",
                "physical_dt",
                "mixed_coefficient_mean_abs",
                "xy_weak_defect",
                "abs_mixed_to_diagonal_sum",
            )
        }
        for r in mixed
    ]
