"""Frozen historical records and mesoscopic binding compatibility."""

import json

import numpy as np
import pytest

from common.experiments.execution import execute_tasks
from common.experiments.plans import freeze_plan
from common.io.json import atomic_json, fingerprint, save_result
from common.runtime.failures import ProtocolError
from theory.runtime.historical import FrozenAuditDataset
from theory.runtime.task_adapters import MeasureAdapter


def test_read_only_legacy_adapter_authenticates_old_envelope_without_rewriting(tmp_path):
    job = {"id": "old", "family": "clocks", "payload": {"id": "old", "kind": "trace"}}
    plan = freeze_plan(tmp_path / "plan.json", {"schema": 2, "jobs": [job], "definitions_sha256": {"old.py": "saved"}})
    atomic_json(
        tmp_path / "binding.json", {"plan_fingerprint": plan["plan_fingerprint"], "source_fingerprint": "source"}
    )
    path = tmp_path / "clocks" / "raw" / "old.json.gz"
    record = {
        "job": job,
        "identity": fingerprint({"task": job, "build": "source"}),
        "build_fingerprint": "source",
        "result": {"value": 3},
    }
    save_result(path, record)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    dataset = FrozenAuditDataset(tmp_path)
    assert dataset.records("clocks") == [record]
    assert all(p.read_bytes() == data for p, data in before.items())
    save_result(path, {**record, "identity": "forged"})
    with pytest.raises(ProtocolError, match="binding mismatch"):
        dataset.records("clocks")


def test_measure_native_batch_uses_same_requests_and_npz_result_fields(tmp_path, monkeypatch):
    from dataclasses import asdict

    import smp_meso_bindings

    from theory.parameters import Parameters
    from theory.runtime.measure import measure_request

    binary = tmp_path / "binary"
    binary.write_bytes(b"unchanged")
    tasks = [
        {
            "id": str(i),
            "kind": "measure",
            "parameters": asdict(Parameters(alpha=0.02, horizon=0.02)),
            "roles": [],
            "request_options": {},
        }
        for i in range(3)
    ]
    calls = []

    def batch(binary_path, requests, workers, **options):
        calls.append(len(requests))
        assert workers == len(requests)
        assert requests == [
            measure_request(t["id"], Parameters(**t["parameters"]))
            for t in tasks
            if t["id"] in {r["request_id"] for r in requests}
        ]
        return [
            {
                "request_id": r["request_id"],
                "result": {
                    "series": {"time": [0, 1], "node_energy": [1, 0], "edge_energy": [1, 0]},
                    "snapshots": {"final_rho": [1], "final_edge": [1]},
                    "diagnostics": {
                        "executed_steps": 1,
                        "converged": False,
                        "max_node_mass_residual": 0,
                        "max_fixed_degree_residual": 0,
                    },
                },
            }
            for r in requests
        ]

    monkeypatch.setattr(smp_meso_bindings, "run_kinetic_batch_parallel", batch)
    monkeypatch.setattr("theory.runtime.task_adapters.run_isolated", lambda f, *args: f(*args))
    execute_tasks(tasks, MeasureAdapter(tmp_path / "measure", binary, binding_batch=True), workers=2)
    assert calls == [2, 1]
    with np.load(tmp_path / "measure" / "raw" / "0.npz", allow_pickle=False) as record:
        assert json.loads(str(record["metadata"])) == tasks[0]
        np.testing.assert_array_equal(record["tau"], [0, 0.02])
