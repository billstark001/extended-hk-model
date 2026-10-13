"""Parameter-bound NPZ measurement serialization with explicit timestamps."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from theory.types import MeasureResult, MeasureTask


def save_result(path: Path, item: MeasureTask, result: MeasureResult) -> None:
    """Persist one sweep result as NPZ, restoring first-step energies while keeping other observables on their original timestamps."""
    series = {key: np.asarray(value) for key, value in result["series"].items()}
    if not {"time", "node_energy", "edge_energy"} <= series.keys():
        raise ValueError(f"{item['id']}: missing energy time series")
    snapshots = result.get("snapshots", {})
    # Retain the first update when the ordinary recorder is coarser. This is
    # essential for large r, where much edge release happens in the first step.
    first = item["parameters"]["dt"]
    if len(series["time"]) > 1 and series["time"][1] > first + 1e-12:
        rho = np.asarray(snapshots["rho"])[1]
        node = 0.5 * np.sum(rho * np.asarray(snapshots["node_potential"])[1])
        edge = np.nansum(rho * np.asarray(snapshots["edge_potential"])[1])
        for key in series:
            if key in ("time", "node_energy", "edge_energy"):
                value = {"time": first, "node_energy": node, "edge_energy": edge}[key]
                series[key] = np.insert(series[key], 1, value)
        # Other observables keep their own regular timestamps; never invent a
        # polarization/homophily value at the inserted energy timestamp.
        series["observable_time"] = np.asarray(result["series"]["time"])
    else:
        series["observable_time"] = series["time"].copy()
    series["tau"] = series["time"] * item["parameters"]["alpha"]
    for key in ("time", "rho", "edge"):
        if key in snapshots:
            series["snapshot_" + key] = np.asarray(snapshots[key])
    for key in ("final_rho", "final_edge"):
        if key in snapshots:
            series[key] = np.asarray(snapshots[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=json.dumps(item, sort_keys=True),
            diagnostics=json.dumps(result["diagnostics"], sort_keys=True),
            **cast(dict[str, Any], series),
        )
    temporary.replace(path)
