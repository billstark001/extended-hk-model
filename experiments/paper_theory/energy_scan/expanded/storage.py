"""Read existing baseline/expanded energy curves without a scan design or solver."""

import json
import math
from pathlib import Path

import numpy as np


def load_shard(path, protocol=None):
    """Read a saved shard, optionally checking its original complete protocol."""
    with np.load(path, allow_pickle=False) as archive:
        if protocol is not None and json.loads(str(archive["protocol"].item())) != protocol:
            raise ValueError(f"incompatible raw shard: {path}")
        return {name: archive[name] for name in archive.files}


def curve_slots(shards):
    """Yield available parameter slots, restoring a saved first physical update.

    Shard keys are (dynamics, alpha). Axes and physical dt come from the saved
    data. No complete historical grid or current executable is required.
    """
    for (dynamics, alpha), shard in shards.items():
        protocol = json.loads(str(shard["protocol"].item())) if "protocol" in shard else {}
        if protocol.get("dynamics", dynamics) != dynamics or not np.isclose(
            protocol.get("alpha", alpha), alpha, rtol=1e-9, atol=1e-12
        ):
            raise ValueError("requested dynamics/alpha differ from the saved protocol")
        first_tau = alpha * protocol.get("dt", 1.0)
        if not np.isfinite(first_tau) or first_tau <= 0:
            raise ValueError("saved alpha and physical dt must be positive and finite")
        for i, epsilon in enumerate(shard["epsilon"]):
            for j, ratio in enumerate(shard["q_over_alpha"]):
                length = int(shard["lengths"][i, j])
                time_key = "tau" if "tau" in shard else "time"
                tau = np.asarray(shard[time_key][i, j, :length], dtype=float)
                if time_key == "time":
                    tau = alpha * tau
                node = np.asarray(shard["node_energy"][i, j, :length], dtype=float)
                edge = np.asarray(shard["edge_energy"][i, j, :length], dtype=float)
                if length < 2 or any(len(a) != length or not np.all(np.isfinite(a)) for a in (tau, node, edge)):
                    raise ValueError("invalid or missing saved energy curve")
                if tau[0] != 0 or np.any(np.diff(tau) <= 0):
                    raise ValueError("saved tau must start at zero and strictly increase")
                if tau[1] > first_tau + 1e-12:
                    if "first_step_delta" not in shard:
                        raise ValueError("coarse curve lacks its saved first-step energy changes")
                    delta = np.asarray(shard["first_step_delta"][i, j], dtype=float)
                    tau = np.insert(tau, 1, first_tau)
                    node = np.insert(node, 1, node[0] + delta[0])
                    edge = np.insert(edge, 1, edge[0] + delta[1])
                else:
                    delta = (
                        np.asarray(shard["first_step_delta"][i, j], dtype=float)
                        if "first_step_delta" in shard
                        else np.asarray((node[1] - node[0], edge[1] - edge[0]))
                    )
                if not np.all(np.isfinite(delta)):
                    raise ValueError("invalid first-step energy changes")
                q = alpha * float(ratio)
                yield {
                    "dynamics": dynamics,
                    "alpha": float(alpha),
                    "epsilon": float(epsilon),
                    "q_over_alpha": float(ratio),
                    "q": 1.0 if math.isclose(q, 1.0, abs_tol=1e-12, rel_tol=0) else q,
                    "converged": bool(shard["converged"][i, j]),
                    "tau": tau,
                    "node_energy": node,
                    "edge_energy": edge,
                    "first_step_delta": delta,
                }


def load_slot(path, *, dynamics, alpha, epsilon, ratio):
    """Select exactly one saved (dynamics, epsilon, alpha, q/alpha) slot."""
    if (
        dynamics not in ("hk", "deffuant")
        or not np.isfinite([alpha, epsilon, ratio]).all()
        or alpha <= 0
        or epsilon <= 0
        or ratio < 0
    ):
        raise ValueError("invalid energy-slot coordinates")
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = (
            path / f"{dynamics}_alpha_{alpha:.12g}.npz",
            path / "raw_shards" / f"{dynamics}_alpha_{alpha:.12g}.npz",
            path / "energy_curve_parameterization.npz",
        )
        present = [p for p in candidates if p.is_file()]
        if len(present) != 1:
            raise ValueError("expected one matching energy archive; specify its NPZ path directly")
        path = present[0]
    # Avoid decoding unrelated potential/terminal-state arrays in a large shard.
    keys = (
        "protocol",
        "epsilon",
        "q_over_alpha",
        "lengths",
        "tau",
        "time",
        "node_energy",
        "edge_energy",
        "first_step_delta",
        "converged",
    )
    with np.load(path, allow_pickle=False) as archive:
        shard = {name: archive[name] for name in keys if name in archive.files}
    if "protocol" not in shard:
        raise ValueError("energy-slot selection requires the saved physical protocol")
    protocol = json.loads(str(shard["protocol"].item()))
    if "dynamics" not in protocol or "alpha" not in protocol:
        raise ValueError("saved energy protocol lacks dynamics/alpha")
    for name, value in (("epsilon", epsilon), ("q_over_alpha", ratio)):
        hits = np.flatnonzero(np.isclose(shard[name], value, rtol=1e-9, atol=1e-12))
        if len(hits) != 1:
            raise ValueError(f"slot {name} must match exactly one saved coordinate")
        index = int(hits[0])
        axis = 0 if name == "epsilon" else 1
        for field in ("lengths", "tau", "time", "node_energy", "edge_energy", "first_step_delta", "converged"):
            if field in shard:
                shard[field] = np.take(shard[field], [index], axis=axis)
        shard[name] = shard[name][[index]]
    slot = next(curve_slots({(dynamics, alpha): shard}))
    return slot, path
