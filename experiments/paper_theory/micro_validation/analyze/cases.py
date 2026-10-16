"""Per-task microscopic/pair field errors and conditional audit diagnostics."""

from __future__ import annotations

import collections
import math

import numpy as np

from common.io.tables import table
from common.statistics import bootstrap_mean
from theory.runtime.dataset import key
from theory.validation import compare_fields, trajectory_gap


def analyze_micro(items, out):
    """Write matched field/energy errors and conditional audit diagnostics from validated records; return the family summary."""
    rows = []
    fields = []
    increments = []
    for rec in items:
        job, value = rec["job"], rec["result"]
        meta = job["meta"]
        pair = value["pair"]
        micro = value["frames"]
        n = trajectory_gap(micro, pair["frames"], "node_energy")
        e = trajectory_gap(micro, pair["frames"], "edge_energy")
        rows.append(
            {
                **meta,
                "id": job["id"],
                "initial_hash": value["initial_hash"],
                "projection_l1": value["projection_l1"],
                "node_energy_gap": n["gap"],
                "edge_energy_gap": e["gap"],
                "edge_initial_match": e["initial_match"],
                "mean_degree": micro[0]["mean_degree"],
                "rewires_per_node": micro[-1]["rewires"] / meta["n"],
                "node_bin_energy_bias": micro[-1]["node_energy"] - micro[-1]["exact_node_energy"],
                "edge_bin_energy_bias": micro[-1]["edge_energy"] - micro[-1]["exact_edge_energy"],
                **compare_fields(micro[-1], pair["frames"][-1]),
            }
        )
        pf = {f["step"]: f for f in pair["frames"] if "rho" in f}
        for f in micro:
            if "rho" in f and f["step"] in pf:
                fields.append(
                    {**meta, "id": job["id"], "tau": meta["alpha"] * f["time"], **compare_fields(f, pf[f["step"]])}
                )
        for a in value.get("audits") or []:
            first, last = (a["pair_one_step"]["frames"][0], a["pair_one_step"]["frames"][-1])
            pred_r = np.array(last["rho"]) - np.array(first["rho"])
            pred_e = np.array(last["edge"]) - np.array(first["edge"])
            mr, me = np.array(a["mean_delta_rho"]), np.array(a["mean_delta_edge"])
            se = np.sqrt(np.asarray(a["variance_delta_edge"]) / a["branches"])
            z = np.divide(abs(me - pred_e), se, out=np.zeros_like(se), where=se > 1e-14)
            frozen = a["frozen"]
            rho = np.array(first["rho"])
            mu = meta["alpha"] * np.array(frozen["mean"])
            pe = np.array(first["edge"]).reshape(meta["b"], meta["b"])
            cross = float(np.sum(pe * np.outer(mu, mu)) / pe.sum())
            active = np.asarray(a["source_counts"]) > 0
            cond = np.asarray(a["mean_increment"])
            second = meta["alpha"] ** 2 * np.asarray(
                frozen["hk_second" if meta["dynamics"] == "hk" else "deffuant_second"]
            )
            increments.append(
                {
                    **meta,
                    "id": job["id"],
                    "tau": meta["alpha"] * a["step"],
                    "branches": a["branches"],
                    "projection_l1": a["projection_l1"],
                    "delta_rho_l1_bias": float(abs(mr - pred_r).sum()),
                    "delta_edge_l1_bias": float(abs(me - pred_e).sum() / job["payload"]["request"]["out_degree"]),
                    "conditional_mean_rms_bias": float(np.sqrt(np.mean((cond[active] - mu[active]) ** 2))),
                    "conditional_second_rms_bias": float(
                        np.sqrt(np.mean((np.asarray(a["second_increment"])[active] - second[active]) ** 2))
                    ),
                    "nonzero_se_edge_z_median": float(np.median(z[se > 1e-14])) if np.any(se > 1e-14) else None,
                    "zero_se_edge_bias_l1": float(abs(me - pred_e)[se <= 1e-14].sum()),
                    "micro_endpoint_product": a["mean_endpoint_product"],
                    "micro_endpoint_product_se": math.sqrt(a["variance_endpoint_product"] / a["branches"]),
                    "pair_endpoint_product": cross,
                    "micro_rewire_fraction": a["rewire_mean"],
                    "micro_rewire_se": math.sqrt(a["rewire_variance"] / a["branches"]),
                    "pair_rewire_fraction": float(meta["q"] * np.dot(rho, frozen["eligibility"])),
                    "micro_eligibility": a["eligibility_mean"],
                    "empty_feed": a["empty_feed_mean"],
                }
            )
    table(out / "trajectory_errors.csv", rows)
    table(out / "distribution_errors_by_time.csv", fields)
    table(out / "conditional_increments.csv", increments)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[
            key(
                r,
                exclude=(
                    "replicate",
                    "id",
                    "initial_hash",
                    "projection_l1",
                    "node_energy_gap",
                    "edge_energy_gap",
                    "edge_initial_match",
                    "mean_degree",
                    "rewires_per_node",
                    "node_bin_energy_bias",
                    "edge_bin_energy_bias",
                    "node_hellinger",
                    "edge_hellinger",
                    "node_w1",
                ),
            )
        ].append(r)
    uncertainty = []
    for group in groups.values():
        for metric in ("node_energy_gap", "edge_energy_gap", "node_hellinger", "edge_hellinger"):
            uncertainty.append(
                {
                    **{
                        k: group[0][k]
                        for k in (
                            "dynamics",
                            "recommender",
                            "zeta",
                            "ratio",
                            "law",
                            "network",
                            "n",
                            "b",
                            "tier",
                            "selection",
                            "alpha",
                            "q",
                            "epsilon",
                            "h",
                            "physical_dt",
                        )
                    },
                    "metric": metric,
                    **bootstrap_mean([r[metric] for r in group if r[metric] is not None]),
                }
            )
    table(out / "replicate_bootstrap.csv", uncertainty)
    return {"trajectories": len(rows), "conditional_checkpoints": len(increments)}
