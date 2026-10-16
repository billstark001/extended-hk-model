"""Matched operator trajectory, final-field and numerical-convergence comparisons."""

from __future__ import annotations

import collections
import math

from common.io.tables import table
from theory.runtime.dataset import key
from theory.validation import compare_fields, trajectory_gap

from .parity import parity


def compare_trace(a, b):
    """Compare matched operator trajectories and final cell masses using common time/grid conventions."""
    result = compare_fields(a["frames"][-1], b["frames"][-1])
    result["node_energy_gap"] = trajectory_gap(a["frames"], b["frames"], "node_energy")["gap"]
    result["edge_energy_gap"] = trajectory_gap(a["frames"], b["frames"], "edge_energy")["gap"]
    result["pathway_gap"] = (
        abs(a["pathway"] - b["pathway"]) if a["pathway"] is not None and b["pathway"] is not None else None
    )
    return result


def analyze_operator(items, out):
    """Write runtime parity, fixed-coefficient convergence and measure/Fokker-Planck discrepancy tables from validated records."""
    pairs = collections.defaultdict(dict)
    convergence = collections.defaultdict(list)
    checks = []
    for rec in items:
        md = rec["job"]["meta"]
        if rec["job"]["payload"]["kind"] == "parity":
            checks.append({"id": rec["job"]["id"], "max_gap": parity(rec["result"])})
            continue
        if md["scan"] in ("validity", "physical_limit"):
            pairs[key(md, ("method",))][md["method"]] = rec
        if md["scan"] == "fixed_coefficients":
            convergence[key(md, ("h", "b"))].append(rec)
    errors = []
    missing = 0
    for pair in pairs.values():
        if set(pair) != {"measure", "fokker_planck"}:
            missing += 1
            continue
        a, b = pair["measure"], pair["fokker_planck"]
        errors.append(
            {
                **a["job"]["meta"],
                "measure_id": a["job"]["id"],
                "fp_id": b["job"]["id"],
                **compare_trace(a["result"], b["result"]),
            }
        )
    mesh = []
    for group in convergence.values():
        for a in group:
            md = a["job"]["meta"]
            for b in group:
                nd = b["job"]["meta"]
                kind = None
                if md["b"] == nd["b"] and math.isclose(md["h"], 2 * nd["h"]):
                    kind = "numerical_step"
                if (
                    md["h"] == nd["h"]
                    and nd["b"] > md["b"]
                    and not any(md["b"] < c["job"]["meta"]["b"] < nd["b"] for c in group)
                ):
                    kind = "spatial_grid"
                if kind:
                    mesh.append(
                        {
                            **md,
                            "reference_b": nd["b"],
                            "reference_h": nd["h"],
                            "comparison": kind,
                            **compare_trace(a["result"], b["result"]),
                        }
                    )
    table(out / "operator_discrepancies.csv", errors)
    table(out / "fixed_coefficient_convergence.csv", mesh)
    table(out / "production_parity.csv", checks)
    return {
        "complete_pairs": len(errors),
        "unpaired_cells": missing,
        "numerical_comparisons": len(mesh),
        "parity_checks": checks,
    }
