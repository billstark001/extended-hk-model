"""Offline controls use and disclose the selected pathway-index convention."""

import json

import numpy as np
import pandas as pd
import pytest

from experiments.paper.pathway_controls import __main__ as cli
from experiments.paper.pathway_controls import study


@pytest.mark.parametrize("endpoint_complement, expected", [(True, .325), (False, .075)])
def test_control_analysis_records_the_selected_endpoint_policy(tmp_path, monkeypatch, endpoint_complement, expected):
    import smp_bindings
    from ehk.micro import observables

    metadata = {"UniqueName": "control", "RNG": {"Seed1": "1"}, "HKParams": {}, "NodeCount": 2,
                "RecsysFactoryType": "none", "RecSysParams": {}}
    raw = tmp_path / "raw/control"
    raw.mkdir(parents=True)
    (raw / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(study, "export_snapshots", lambda path: {"snapshots": [{}, {}]})
    monkeypatch.setattr(study, "read_plan", lambda path: {"tasks": [{"id": "one", "coordinates": {}}],
                        "plan_fingerprint": "plan", "protocol": {"horizon": 8}})
    monkeypatch.setattr(study, "ARMS", ["full"])
    monkeypatch.setattr(study, "arm_metadata", lambda task, arm: metadata)

    class Record:
        max_step = 8

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(smp_bindings, "RawSimulationRecord", Record)
    monkeypatch.setattr(observables, "get_indices", lambda record: (
        np.array([0, 8]), np.array([.25, .5]), np.array([.3, .5]), np.array([.4, .6]),
    ))
    study.analyze(tmp_path, endpoint_complement=endpoint_complement)
    saved = pd.read_csv(tmp_path / "analysis/pathway_integrals.csv").iloc[0]
    assert saved["I_w"] == pytest.approx(expected)
    assert bool(saved["endpoint_complement"]) == endpoint_complement
    receipt = json.loads((tmp_path / "analysis/receipt.json").read_text())
    assert receipt["endpoint_complement"] == endpoint_complement
    assert receipt["metric_protocol"]["endpoint_complement"] == endpoint_complement


@pytest.mark.parametrize("flag, expected", [([], True), (["--no-endpoint-complement"], False)])
def test_control_cli_defaults_to_complement_and_can_disable_it(tmp_path, monkeypatch, flag, expected):
    calls = []
    monkeypatch.setattr(cli, "analyze", lambda *args, **kwargs: calls.append((args, kwargs)))
    cli.main(["analyze", "--directory", str(tmp_path), *flag])
    assert calls == [((tmp_path,), {"endpoint_complement": expected})]
