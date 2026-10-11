"""Validation manuscript summaries assembled from study-owned evidence reviews."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.paper_theory.reporting import read_csv, summarize, write_csv

from ...kernel_validation.analyze.review import review_kernel
from ...micro_validation.analyze.review import review_micro
from ...operator_validation.analyze.review import review_operator
from ...release_clocks.analyze.review import review_clocks
from ..figures.validation import plots


def review(root, out):
    """Write micro/clock/operator/kernel report tables and figures from a frozen validation run; unavailable selected examples stay absent."""
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "root": str(root),
        "review_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_binding": json.loads((root / "binding.json").read_text()),
        "analysis_table_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.glob("*/analysis/*.csv")
            if not p.name.startswith(".")
        },
    }
    micro = read_csv(root, "micro", "trajectory_errors.csv")
    clocks = read_csv(root, "clocks", "direct_clocks.csv")
    operator = read_csv(root, "operator", "operator_discrepancies.csv")
    numerical = read_csv(root, "operator", "fixed_coefficient_convergence.csv")
    kernel = read_csv(root, "kernel", "finite_list_audit.csv")
    metrics = ("node_w1", "node_hellinger", "edge_hellinger", "node_energy_gap", "edge_energy_gap")
    write_csv(
        out / "micro_groups.csv",
        summarize(micro, ("tier", "holdout", "law", "dynamics", "recommender", "zeta", "ratio", "b"), metrics),
    )
    summary["micro_by_ratio"] = summarize([r for r in micro if r["tier"] == "core"], ("ratio",), metrics)
    summary["heldout_initial"] = summarize(
        [r for r in micro if r["tier"] == "heldout" and r.get("holdout") != "parameter"], ("law",), metrics
    )
    summary["heldout_parameter"] = summarize([r for r in micro if r.get("holdout") == "parameter"], (), metrics)
    selected = review_micro(root, out, summary, micro)
    review_clocks(root, out, summary, clocks)
    review_operator(out, summary, operator, numerical, metrics)
    review_kernel(root, out, summary, kernel)
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    plots(out, micro, clocks, selected)
    print(
        json.dumps(
            {
                "out": str(out),
                "micro": len(micro),
                "clocks": len(clocks),
                "operator_pairs": len(operator),
                "kernel_source_rows": len(kernel),
                "primary_clock_verified": summary["clock_primary_verified"],
                "energy_only_posthoc_added": summary["energy_only_posthoc_added"],
                "crossings_over_missing_points": summary["clock_crossings_across_missing"],
            },
            ensure_ascii=False,
        )
    )
