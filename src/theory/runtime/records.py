"""Microscopic record fields and verified deterministic pair references."""

from pathlib import Path

from common.io.json import fingerprint, sha
from common.io.records import promote_arrays, read_record

ARRAY_FIELDS = {"axis", "rho", "edge", "velocity", "node_potential", "edge_potential", "polarization", "homophily"}


def scientific_arrays(value):
    """Promote legacy state vectors to float64 without changing scalar metadata."""
    return promote_arrays(value, ARRAY_FIELDS)


def read_micro(path, *, record=None):
    """Resolve a shared pair in memory; optionally reuse an already decoded record."""
    path = Path(path)
    record = read_record(path) if record is None else record
    reference = record["result"].get("pair_ref")
    if reference is not None:
        if len(reference["key"]) != 64 or any(c not in "0123456789abcdef" for c in reference["key"]):
            raise ValueError("invalid shared pair key")
        pair_path = path.parent.parent / "pairs" / (reference["key"] + ".msgpack.lz4")
        if pair_path.resolve().parent != (path.parent.parent / "pairs").resolve():
            raise ValueError("pair reference escapes the run")
        if sha(pair_path) != reference["sha256"]:
            raise ValueError("shared pair hash mismatch")
        pair = read_record(pair_path)
        identity = {k: v for k, v in pair["task"].items() if k != "id"}
        if pair["task"]["id"] != reference["key"] or fingerprint(identity) != reference["key"]:
            raise ValueError("shared pair identity mismatch")
        if pair["result"]["initial_hash"] != record["result"]["initial_hash"]:
            raise ValueError("shared pair initial state mismatch")
        for name in ("parameters", "initial_seed", "request_options", "selection"):
            if pair["task"].get(name) != record["task"].get(name):
                raise ValueError(f"shared pair {name} mismatch")
        trace = dict(pair["result"]["pair"])
        if "pair_diagnostics" in record["result"]:
            trace["diagnostics"] = record["result"]["pair_diagnostics"]
        record["result"]["pair"] = trace
    return record
