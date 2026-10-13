"""Type contracts reject malformed task/frame dictionaries before execution."""

import importlib.util
import subprocess
import sys
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from common.io.records import content_fingerprint
from experiments.paper_theory.evidence.analyze import compact_record
from theory.parameters import Parameters
from theory.types import ParametersData, ResolvedMicroRecord


def test_wire_parameter_schema_tracks_the_parameter_dataclass():
    assert set(ParametersData.__annotations__) == {field.name for field in fields(Parameters)}


def test_typed_compaction_retains_snapshots_endpoints_and_exact_energy_samples():
    record = cast(
        ResolvedMicroRecord,
        {
            "task": {"id": "fixture"},
            "result": {
                "frames": [
                    {
                        "step": i,
                        "time": float(i),
                        "exact_node_energy": 1 - i / 3,
                        "exact_edge_energy": 1 - i / 4,
                        "rho": [0.5, 0.5],
                        "exact_variance": 0.25,
                        **({"edge": [0.0, 0.0, 0.0, 1.0]} if i == 0 else {}),
                    }
                    for i in range(3)
                ],
                "pair": {
                    "frames": [
                        {
                            "step": i,
                            "time": float(i),
                            "node_energy": 1 - i / 3,
                            "edge_energy": 1 - i / 4,
                            "rho": [0.5, 0.5],
                            "budget_motion": float(i),
                            **({"edge": [0.0, 0.0, 0.0, 1.0]} if i == 0 else {}),
                        }
                        for i in range(3)
                    ]
                },
            },
        },
    )
    before = deepcopy(record)
    fingerprint = content_fingerprint(before["result"]["pair"]["frames"])
    compact = compact_record(record)["result"]
    assert compact["pair_frames_fingerprint"] == fingerprint
    assert set(compact["frames"][1]) == {"time", "step", "exact_node_energy", "exact_edge_energy"}
    assert set(compact["pair"]["frames"][1]) == {"time", "step", "node_energy", "edge_energy"}
    for i in range(3):
        assert compact["frames"][i]["exact_node_energy"] == before["result"]["frames"][i]["exact_node_energy"]
        assert compact["pair"]["frames"][i]["edge_energy"] == before["result"]["pair"]["frames"][i]["edge_energy"]
    assert compact["frames"][-1]["exact_variance"] == 0.25
    assert compact["pair"]["frames"][-1]["budget_motion"] == 2.0
    assert np.array_equal(compact["frames"][0]["edge"], before["result"]["frames"][0]["edge"])


def test_static_contracts_reject_bad_seeds_missing_fields_and_misspelled_energies(tmp_path):
    if importlib.util.find_spec("mypy") is None:
        pytest.skip("install the optional typecheck dependencies to exercise static negative fixtures")
    fixture = tmp_path / "invalid_contracts.py"
    fixture.write_text(
        """from typing import cast
from theory.types import MicroTask, MicroFrame, ParametersData, PairReference

parameters = cast(ParametersData, {})
bad_seed: MicroTask = {
    "id": "same", "kind": "micro", "parameters": parameters,
    "request_options": {}, "roles": [], "initial": 0, "branch": 0,
    "initial_seed": 1, "process_seed": "resampled", "selection": "weighted",
}
missing_seed: MicroTask = {
    "id": "same", "kind": "micro", "parameters": parameters,
    "request_options": {}, "roles": [], "initial": 0, "branch": 0,
    "initial_seed": 1, "selection": "weighted",
}
bad_frame: MicroFrame = {
    "step": 0, "time": 0.0, "exact_node_energy": 1.0,
    "exact_edge_energy": 1.0, "node_enrgy": 1.0,
}
bad_reference: PairReference = {"key": "same", "sha256": 123}
"""
    )
    checked = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-incremental", "--show-error-codes", str(fixture)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 1, checked.stdout + checked.stderr
    assert '"process_seed"' in checked.stdout
    assert '"node_enrgy"' in checked.stdout
    assert '"sha256"' in checked.stdout
    assert "[typeddict-item]" in checked.stdout
    assert "Found 4 errors" in checked.stdout
