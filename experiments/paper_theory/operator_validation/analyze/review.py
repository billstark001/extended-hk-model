"""Operator approximation and numerical-convergence evidence for manuscript summaries."""

from __future__ import annotations

from experiments.paper_theory.reporting import summarize, write_csv


def review_operator(out, summary, operator, numerical, metrics):
    """Aggregate saved operator/numerical discrepancies into manuscript tables and the supplied summary."""
    write_csv(
        out / "operator_groups.csv",
        summarize(
            operator, ("scan", "dynamics", "alpha", "recommender", "zeta", "physical_dt"), metrics + ("pathway_gap",)
        ),
    )
    write_csv(
        out / "numerical_groups.csv", summarize(numerical, ("comparison", "alpha", "h", "b", "reference_b"), metrics)
    )
    summary["operator_by_alpha"] = summarize(
        [r for r in operator if r["scan"] == "validity"], ("alpha",), metrics + ("pathway_gap",)
    )
