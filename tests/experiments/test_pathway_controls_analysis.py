"""Typed offline controls disclose conventions and resume verified arm results."""
import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
from common.io.json import atomic_json
from common.storage.documents import DocumentStore
from experiments.paper.pathway_controls import __main__ as cli, study
from experiments.paper.pathway_controls.design import make_plan, arm_metadata
from experiments.paper.pathway_controls.contracts import SnapshotIndex, SnapshotEntry, SnapshotInput


@pytest.mark.parametrize("endpoint_complement,expected", [(True,.325),(False,.075)])
def test_control_analysis_records_the_selected_endpoint_policy_and_resumes(tmp_path, monkeypatch, endpoint_complement, expected):
    import smp_bindings
    from ehk.micro import observables
    plan = replace(make_plan(cases=["baseline"], replicates=1, horizon=8, node_count=8, follow_count=2), plan_fingerprint="plan")
    metadata = arm_metadata(plan.tasks[0], "full")
    raw = tmp_path / "raw" / metadata.UniqueName
    raw.mkdir(parents=True)
    atomic_json(raw / "metadata.json", metadata)
    monkeypatch.setattr(study, "export_snapshots", lambda path: SnapshotIndex(1,"plan",()))
    monkeypatch.setattr(study, "read_plan", lambda path: plan)
    monkeypatch.setattr(study, "ARMS", ["full"])
    calls = []
    class Record:
        max_step = 8
        is_finished = is_sanitized = True
        def __init__(self,*args):
            pass
        def __enter__(self):
            calls.append("compute")
            return self
        def __exit__(self,*args):
            pass
    monkeypatch.setattr(smp_bindings,"RawSimulationRecord",Record)
    monkeypatch.setattr(observables,"get_indices",lambda record: (
        np.array([0,8]),np.array([.25,.5]),np.array([.3,.5]),np.array([.4,.6])))
    first = study.analyze(tmp_path, endpoint_complement=endpoint_complement)
    assert first.arms == 1
    assert calls == ["compute"]
    saved = pd.read_csv(tmp_path / "analysis/pathway_integrals.csv").iloc[0]
    assert saved["I_w"] == pytest.approx(expected)
    assert bool(saved["endpoint_complement"]) == endpoint_complement
    with DocumentStore(tmp_path / "analysis/analysis.sqlite",readonly=True) as store:
        receipt = next(store.values("controls.receipts"))
    assert receipt["endpoint_complement"] == endpoint_complement
    assert receipt["metric_protocol"]["endpoint_complement"] == endpoint_complement


@pytest.mark.parametrize("flag,expected", [([],True),(["--no-endpoint-complement"],False)])
def test_control_cli_defaults_to_complement_and_can_disable_it(tmp_path,monkeypatch,flag,expected):
    calls=[]
    monkeypatch.setattr(cli,"analyze",lambda *args,**kwargs:calls.append((args,kwargs)))
    cli.main(["analyze","--directory",str(tmp_path),*flag])
    assert calls==[((tmp_path,),{"endpoint_complement":expected})]


def test_snapshot_plot_reads_analyzed_sqlite_with_raw_checkpoints_disconnected(tmp_path,monkeypatch):
    plan=replace(make_plan(cases=["baseline"],replicates=1,horizon=8,node_count=8,follow_count=2),plan_fingerprint="frozen")
    monkeypatch.setattr(study,"read_plan",lambda path:plan)
    monkeypatch.setattr(study,"ARMS",["full"])
    monkeypatch.setattr(study,"export_snapshots",lambda *args:pytest.fail("plot started raw analysis"))
    monkeypatch.setattr(study,"_physical",lambda *args:pytest.fail("plot opened raw checkpoint"))
    with DocumentStore(tmp_path / "analysis.sqlite") as store:
        store.put("controls.index","frozen",SnapshotIndex(1,"frozen",(
            SnapshotEntry("pair","baseline",0,"full",0,"unused","source-sha","state","state-fp"),)))
        store.put("controls.snapshots","state",SnapshotInput(1,0,"HK",(0,1),(-.3,.3),((0,1),),()))
    assert (study.plot_snapshots(tmp_path) / "pair.png").stat().st_size >100
