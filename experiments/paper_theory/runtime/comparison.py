"""Frozen task/build identity checks and matched baseline-prefix comparisons."""

from __future__ import annotations

from common.io.tables import table
from theory.runtime.historical import FrozenAuditDataset


def physical_key(job):
    """Return the physical, numerical and seed comparison tuple, excluding horizons, run controls and request IDs."""
    m, p = job["meta"], job["payload"]
    r = p["request"]
    return tuple(
        m[k]
        for k in (
            "dynamics",
            "recommender",
            "zeta",
            "alpha",
            "ratio",
            "epsilon",
            "n",
            "b",
            "h",
            "law",
            "network",
            "selection",
        )
    ) + (
        r["resolution"]["opinion_quadrature_points"],
        p["init_seed"],
        p["process_seed"] if job["family"] == "micro" else None,
    )


def baseline_prefixes(items, baseline, binding, out):
    """Write checks comparing matching extended-run prefixes with completed baseline trajectories under the same build binding."""
    path = baseline / "plan.json"
    if not path.exists():
        return {"available": False}
    dataset = FrozenAuditDataset(baseline)
    previous, previous_binding = dataset.plan, dataset.binding
    if previous_binding["source_fingerprint"] != binding["source_fingerprint"]:
        return {"available": True, "same_solver_source": False, "comparison": "skipped: changed solver source"}

    def physical_key(job):
        md, req = job["meta"], job["payload"]["request"]
        return tuple(md[k] for k in ("dynamics", "recommender", "zeta", "alpha", "ratio", "epsilon", "b")) + (
            req["resolution"]["opinion_quadrature_points"],
        )

    index = {physical_key(j): j for j in previous["jobs"] if j["family"] == "clocks"}
    comparisons = []
    for record in items:
        old = index.get(physical_key(record["job"]))
        if old is None:
            continue
        p = baseline / "clocks" / "raw" / (old["id"] + ".json.gz")
        if not p.exists():
            continue
        saved = dataset.read(old)
        frames = {f["step"]: f for f in record["result"]["frames"]}
        gaps = [
            abs(f[k] - frames[f["step"]][k])
            for f in saved["result"]["frames"]
            if f["step"] in frames
            for k in ("node_energy", "edge_energy", "budget_rewire", "budget_motion")
        ]
        gap = max(gaps) if gaps else None
        if gap is not None and gap > 1e-10:
            raise ValueError("old/new clock prefix changed despite matching physical parameters and solver source")
        comparisons.append({"id": record["job"]["id"], "baseline_id": old["id"], "max_scalar_prefix_gap": gap})
    table(out / "baseline_prefix_checks.csv", comparisons)
    return {"available": True, "same_solver_source": True, "checked_clock_prefixes": len(comparisons)}
