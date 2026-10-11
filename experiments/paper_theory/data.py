"""Small explicit source readers, without importing historical research scripts."""

import csv
import json
from pathlib import Path

import numpy as np

from .paths import CLOCK_INPUT, SCAN_INPUT, input_root

MANIFEST = Path(__file__).with_name("sources.json")


def sources(root=None, clock_root=None):
    """Return named dataset paths from sources.json using independently configured scan and clock input roots."""
    manifest = json.loads(MANIFEST.read_text())
    roots = {"scan": input_root(root, SCAN_INPUT), "clocks": input_root(clock_root, CLOCK_INPUT)}
    return {key: (roots[item["root"]] / item["path"]).resolve() for key, item in manifest["datasets"].items()}


def read_csv(path):
    """Read CSV records as dictionaries of strings, preserving field names and row order."""
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    """Write nonempty rows with their union of fields, creating parents; reject empty input."""
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows for {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def case_key(row):
    """Return a stable dynamics/alpha/epsilon/r key, rounding alpha and the rate ratio for cross-table joins."""
    return (
        row["dynamics"],
        round(float(row["alpha"]), 12),
        round(float(row["epsilon"]), 12),
        round(float(row.get("r", row.get("q_over_alpha"))), 9),
    )


def raw_release_cases(root=None):
    """Yield copied historical energy trajectories in tau units, restoring saved first-step drops when recording was coarse."""
    files = sorted(sources(root)["release_raw"].glob("*.npz"))
    if not files:
        raise FileNotFoundError("no release shards; set --scan-input or EHK_RELEASE_SCAN_INPUT_DIR")
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            protocol = json.loads(str(archive["protocol"]))
            alpha = float(protocol.get("alpha", path.stem.split("_alpha_")[-1]))
            for i, epsilon in enumerate(archive["epsilon"]):
                for j, ratio in enumerate(archive["q_over_alpha"]):
                    length = int(archive["lengths"][i, j])
                    if length < 2:
                        raise ValueError(f"invalid trajectory length: {path}:{i}:{j}")
                    time = archive["tau"][i, j, :length].copy()
                    node = archive["node_energy"][i, j, :length].copy()
                    edge = archive["edge_energy"][i, j, :length].copy()
                    first_tau = alpha * float(protocol.get("dt", 1))
                    if "first_step_delta" in archive and time[1] > first_tau + 1e-12:
                        time = np.insert(time, 1, first_tau)
                        node = np.insert(node, 1, node[0] + archive["first_step_delta"][i, j, 0])
                        edge = np.insert(edge, 1, edge[0] + archive["first_step_delta"][i, j, 1])
                    yield {
                        "dynamics": protocol.get("dynamics", path.name.split("_")[0]),
                        "epsilon": float(epsilon),
                        "alpha": alpha,
                        "r": float(ratio),
                        "recommender": protocol["recommender"],
                        "bins": protocol["grid_size"],
                        "degree": protocol["mean_degree"],
                        "recommendations": protocol["recommendation_count"],
                        "dt": protocol["dt"],
                        "noise_diffusion": protocol["noise_diffusion"],
                        "confidence_mode": protocol["confidence_mode"],
                        "initial_law": "uniform[-1,1]",
                        "converged": bool(archive["converged"][i, j]),
                        "tau": time,
                        "node_energy": node,
                        "edge_energy": edge,
                        "source": path.name,
                    }
