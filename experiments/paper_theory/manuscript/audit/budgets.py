"""Audit discrete energy budgets against frozen raw trajectories and the final manuscript claims."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from ...paths import REVISION_INPUT
from ..paths import FIGURES_OUTPUT


def main():
    """Recompute discrete channel/mixed budget evidence from external frozen runs and write the manuscript budget audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=REVISION_INPUT.resolve())
    parser.add_argument("--output", type=Path, default=FIGURES_OUTPUT)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve() / "revision_core_v3"
    plan = json.loads((root / "plan.json").read_text())
    status = {r["id"]: r for r in json.loads((root / "status.json").read_text())["records"]}
    records = []
    for job in plan["jobs"]:
        if job["family"] != "clocks":
            continue
        path = root / "clocks/raw" / (job["id"] + ".json.gz")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == status[job["id"]]["sha256"]
        data = json.loads(gzip.decompress(raw))
        assert data["job"] == job
        frames = data["result"]["frames"]
        increments = [b["budget_rewire"] - a["budget_rewire"] for a, b in zip(frames, frames[1:])]
        records.append(
            {
                "id": job["id"],
                "raw_sha256": digest,
                "maximum_saved_interval_rewire_increment": max(increments),
                "positive_saved_intervals_above_1e-12": sum(v > 1e-12 for v in increments),
                "maximum_cumulative_rewire": max(r["budget_rewire"] for r in frames),
            }
        )
    assert len(records) == 202
    result = {
        "scope": "202 original clocks; saved cumulative-budget differences, not every physical substep",
        "sign_tolerance": 1e-12,
        "records": records,
        "max_saved_interval_increment": max(r["maximum_saved_interval_rewire_increment"] for r in records),
        "positive_interval_count": sum(r["positive_saved_intervals_above_1e-12"] for r in records),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rewiring_sign_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("raw budget records verified", len(records), "positive saved intervals", result["positive_interval_count"])


if __name__ == "__main__":
    main()
