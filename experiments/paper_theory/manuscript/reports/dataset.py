"""Bind ensemble manuscript evidence to complete frozen jobs and baseline records."""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np

from common.io.json import fingerprint, sha
from common.io.tables import table
from experiments.paper_theory.manuscript.reports.tables import number, read_table
from experiments.paper_theory.micro_validation.analyze.ensembles import finite_micro_row
from experiments.paper_theory.release_clocks.analyze.ensembles import clock_row
from experiments.paper_theory.runtime.comparison import physical_key
from theory.runtime.historical import FrozenAuditDataset


def gather(root, baseline, out):
    """Validate complete ensemble records and recompute the saved scalar summaries.

    Require the baseline to share the source binding, verify raw hashes/identities
    and compare overlapping physical prefixes. Write audit tables and return
    (plan, reduced_micro_records, reduced_clock_records, audit_summary).
    """
    dataset = FrozenAuditDataset(root)
    plan, binding, expected = dataset.plan, dataset.binding, dataset.expected
    status = json.loads((root / "status.json").read_text())
    assert status["binding"] == binding
    assert len(status["records"]) == len(expected)
    assert len({r["id"] for r in status["records"]}) == len(expected)
    previous_dataset = FrozenAuditDataset(baseline)
    previous, previous_binding = previous_dataset.plan, previous_dataset.binding
    assert previous_binding["source_fingerprint"] == binding["source_fingerprint"]
    old_index = {(j["family"], physical_key(j)): j for j in previous["jobs"] if j["family"] in ("micro", "clocks")}
    prior_status = {r["id"]: r for r in json.loads((baseline / "status.json").read_text())["records"]}
    existing = {
        f: {r["id"]: r for r in read_table(root / "analysis" / f / name)}
        for f, name in (("micro", "trajectory_errors.csv"), ("clocks", "direct_clocks.csv"))
    }
    micro, clock, overlap, hashes = [], [], [], {}
    bytes_total = 0
    for count, state in enumerate(status["records"], 1):
        assert state["status"] in ("complete", "cached")
        job = expected[state["id"]]
        path = root / job["family"] / "raw" / (job["id"] + ".json.gz")
        assert path.resolve() == Path(state["path"]).resolve()
        digest = sha(path)
        if "sha256" in state:
            assert digest == state["sha256"], "raw SHA256 mismatch"
        hashes[job["id"]] = digest
        bytes_total += path.stat().st_size
        record = dataset.read(job)
        m, v = job["meta"], record["result"]
        if job["family"] == "micro":
            row = finite_micro_row(record, plan["analysis_protocol"])
            scalar_keys = (
                "node_w1",
                "node_energy_gap",
                "edge_energy_gap",
                "node_micro_tau50",
                "edge_micro_tau50",
                "node_pair_tau50",
                "edge_pair_tau50",
                "final_micro_node_energy",
                "final_pair_node_energy",
            )
            reduced = {
                "meta": m,
                "id": job["id"],
                "initial_hash": v["initial_hash"],
                "times": np.asarray([f["time"] for f in v["frames"]]),
                "rho": v["frames"][-1]["rho"],
                "pair_rho": v["pair"]["frames"][-1]["rho"],
            }
            for channel in ("node", "edge"):
                reduced[channel] = np.asarray([f["exact_" + channel + "_energy"] for f in v["frames"]])
                reduced[channel + "_pair"] = np.asarray([f[channel + "_energy"] for f in v["pair"]["frames"]])
            micro.append(reduced)
        else:
            row = clock_row(record, plan["analysis_protocol"])
            scalar_keys = (
                "actual_tau",
                "rho_T50",
                "edge_T50",
                "delta50",
                "rho_tail_relative_change",
                "edge_tail_relative_change",
                "budget_residual",
            )
            assert row["energy_confirmed"] == (existing["clocks"][job["id"]]["energy_confirmed"] == "True")
            assert row["joint_confirmed"] == (existing["clocks"][job["id"]]["joint_confirmed"] == "True")
            clock.append(
                {
                    "meta": m,
                    "id": job["id"],
                    "diagnostics": v["diagnostics"],
                    "row": row,
                    "times": np.asarray([f["time"] for f in v["frames"]]),
                    **{c: np.asarray([f[c + "_energy"] for f in v["frames"]]) for c in ("node", "edge")},
                }
            )
        for key in scalar_keys:
            a, b = row[key], number(existing[job["family"]][job["id"]], key)
            assert (a is None and b is None) or (
                a is not None and b is not None and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)
            ), f"primary table differs from raw: {key}"
        old_job = old_index.get((job["family"], physical_key(job)))
        # Match micro comparisons only at the same declared horizon.
        if old_job and (
            job["family"] == "clocks" or old_job["payload"]["request"]["steps"] == job["payload"]["request"]["steps"]
        ):
            old_path = baseline / job["family"] / "raw" / (old_job["id"] + ".json.gz")
            old = previous_dataset.read(old_job)
            if "sha256" in prior_status.get(old_job["id"], {}):
                assert sha(old_path) == prior_status[old_job["id"]]["sha256"]
            frames = {f["step"]: f for f in v["frames"]}
            names = (
                ("exact_node_energy", "exact_edge_energy")
                if job["family"] == "micro"
                else ("node_energy", "edge_energy", "budget_rewire", "budget_motion")
            )
            gap = max(
                abs(f[k] - frames[f["step"]][k]) for f in old["result"]["frames"] if f["step"] in frames for k in names
            )
            assert gap < 1e-10, "old/new matched trajectory changed"
            overlap.append(
                {"family": job["family"], "id": job["id"], "baseline_id": old_job["id"], "max_scalar_gap": gap}
            )
        if count % 50 == 0:
            print(f"review audit {count}/{len(expected)}", flush=True)
    table(out / "baseline_overlap_checks.csv", overlap)
    audit = {
        "plan_fingerprint": plan["plan_fingerprint"],
        "binding": binding,
        "expected": len(expected),
        "complete": len(micro) + len(clock),
        "raw_sha_checks": len(hashes),
        "raw_bytes": bytes_total,
        "wall_minutes": status["wall_seconds"] / 60,
        "raw_sha_manifest_fingerprint": fingerprint(hashes),
        "primary_scalar_tables_recomputed": True,
        "baseline_overlaps": dict(collections.Counter(r["family"] for r in overlap)),
        "max_baseline_scalar_gap": max((r["max_scalar_gap"] for r in overlap), default=None),
        "recorded_primary_analysis_sources_match": {
            name: sha(Path(__file__).resolve().parents[2] / name) == digest
            for name, digest in json.loads((root / "analysis" / "summary.json").read_text())["analysis_sources"].items()
        },
    }
    return plan, micro, clock, audit
