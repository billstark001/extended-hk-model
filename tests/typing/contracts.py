"""Positive static fixtures: valid wire dictionaries retain their existing shape."""

from dataclasses import asdict
from typing import cast

import numpy as np

from common.types import PendingCommit, RecordHeader, RunEvent
from experiments.paper_theory.evidence.design import design
from experiments.paper_theory.types import MainPlan
from theory.parameters import Parameters
from theory.types import MicroFrame, MicroTask, PairReference, ParametersData, ResolvedMicroRecord, StoppingOptions

parameters = cast(ParametersData, asdict(Parameters()))
task: MicroTask = {
    "id": "fixture",
    "kind": "micro",
    "parameters": parameters,
    "request_options": {"snapshot_taus": [0.0, 1.0]},
    "roles": ["matched"],
    "initial": 0,
    "branch": 0,
    "initial_seed": 17,
    "process_seed": 24,
    "selection": "weighted",
}
frame: MicroFrame = {"step": 0, "time": 0.0, "exact_node_energy": 1.0, "exact_edge_energy": 1.0}
snapshot: MicroFrame = {**frame, "projected": True, "rho": np.array([0.5, 0.5], dtype=np.float64)}
reference: PairReference = {"key": "0" * 64, "sha256": "1" * 64}
record: ResolvedMicroRecord = {
    "task": task,
    "result": {
        "initial_hash": "same-state",
        "projection_l1": 0.0,
        "frames": [frame, snapshot],
        "pair_ref": reference,
        "pair": {
            "frames": [{"step": 0, "time": 0.0, "node_energy": 1.0, "edge_energy": 1.0}],
            "diagnostics": {"max_node_mass_residual": 0.0, "max_fixed_degree_residual": 0.0},
        },
    },
}
header: RecordHeader[MicroTask] = {"format": "ehk-record", "schema": 2, "task": task}
pending: PendingCommit = {"destination": "raw/fixture", "source": "staged", "sha256": None}
event: RunEvent = {"time": 0.0, "task": "fixture", "state": "retrying", "attempt": 2}
stopping: StoppingOptions = {"mode": "state_and_energy", "minimum_steps": 120}
plan: MainPlan = design("main_smoke")  # Only constructs a parameter plan; never runs a model.
