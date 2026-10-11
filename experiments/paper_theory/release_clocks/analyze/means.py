"""Finite-window observed release means and separately identified fit extrapolations."""

import json

import numpy as np

from experiments.paper_theory.data import case_key, raw_release_cases, read_csv, sources, write_csv
from experiments.paper_theory.scaling.analyze import scaling_rows
from theory.release import release_clock
from theory.release.fitted_means import fitted_clocks


def summarize_case(case):
    """Return observed finite-window integral clocks and plateau/rebound diagnostics for one case, retaining physical metadata."""
    row = {key: case[key] for key in ("dynamics", "epsilon", "alpha", "r", "converged")}
    row.update(
        {
            key: case[key]
            for key in (
                "recommender",
                "steepness",
                "quadrature",
                "degree",
                "recommendations",
                "bins",
                "population",
                "dt",
                "noise_diffusion",
                "confidence_mode",
                "initial_law",
            )
            if key in case
        }
    )
    row["time_units"] = "tau=alpha*t"
    row["source"] = case.get("source", "new_measure_run")
    for name in ("node", "edge"):
        clock = release_clock(case["tau"], case[f"{name}_energy"], converged=case["converged"])
        row.update(
            {
                f"{name}_{key}": getattr(clock, key)
                for key in ("plateau", "amplitude", "area", "t50", "rebound", "negative_area", "status")
            }
        )
    row["horizon_tau"] = float(case["tau"][-1])
    row["observed_edge_to_node"] = row["edge_area"] / row["node_area"] if row["node_area"] > 0 else np.nan
    return row


def existing_release(root, scan=None, clocks=None):
    """Write observed and fitted mean-clock tables from archived inputs, plus scaling and optional reference-comparison summaries."""
    locations = sources(scan, clocks)
    nf = {
        case_key(r): r
        for r in read_csv(locations["node_fits"])
        if r["energy"] == "node_energy" and r["model"] == "gompertz_hazard_2"
    }
    ef = {case_key(r): r for r in read_csv(locations["edge_fits"])}
    rows, cases = [], []
    for case in raw_release_cases(scan):
        row = summarize_case(case)
        key = case_key(case)
        row["fit_status"] = "missing"
        if key in nf and key in ef:
            try:
                row.update(fitted_clocks(case, nf[key], ef[key]))
                row["fit_status"] = "available"
            except ValueError as error:
                row["fit_status"] = str(error)
        rows.append(row)
        cases.append(case)
    write_csv(root / "tables" / "mean_release.csv", rows)
    write_csv(root / "tables" / "existing_scaling.csv", scaling_rows(cases))
    summary = {
        "trajectories": len(rows),
        "converged": sum(r["converged"] for r in rows),
        "fits_available": sum(r["fit_status"] == "available" for r in rows),
        "units": "tau=alpha*t",
        "mean_definition": "area under normalized remaining energy; finite observed window",
        "plateau": "tail median; see per-row status",
        "fitted_mean": "infinite integral of empirical survival fit",
        "positive_r_converged": sum(r["converged"] and r["r"] > 0 for r in rows),
    }
    reference_path = locations["mean_release_reference"]
    summary["reference_available"] = reference_path.is_file()
    reference = {case_key(r): r for r in read_csv(reference_path)} if reference_path.is_file() else {}
    errors = []
    for row in rows:
        old = reference.get(case_key(row))
        if old and "fitted_mean_edge" in row:
            errors.append(abs(row["fitted_mean_edge"] / float(old["mean_edge"]) - 1))
    summary["reference_fitted_mean_relative_error_max"] = max(errors, default=None)
    (root / "tables" / "release_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
