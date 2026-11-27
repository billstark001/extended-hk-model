"""Raw collection uses canonical indices and preserves missing-media status."""
import json
import subprocess
from dataclasses import replace
import networkx as nx
import numpy as np
import numpy.testing
import pandas as pd
import pytest
from common.io.json import atomic_json,canonical,sha
from ehk.domain import rows_frame,parse_metadata,RecommenderParams
from ehk.metrics.evaluation import MeasurementEvaluator
from experiments.paper.metric_comparison import core,raw
from experiments.paper.metric_comparison.contracts import PreparedPoint
from experiments.paper.recommendation_shared.storage import write_table
from tests.experiments import numerical_fixtures as f

@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    monkeypatch.setattr(subprocess,"Popen",lambda *args,**kwargs:pytest.fail("offline analysis started a native process"))


def _prepared(tmp_path,monkeypatch,*,available):
    preparation=tmp_path/"preparation"
    preparation.mkdir()
    atomic_json(preparation/"preparation.json",f.prepared())
    monkeypatch.setattr(core,"authenticate_preparation",lambda _:f.prepared())
    directory=tmp_path/"disk/raw/task/late/hold"
    if available:
        directory.mkdir(parents=True)
        atomic_json(directory/"metadata.json",f.metadata())
        for name in ("trajectory.json","events.db","graph-000000.msgpack","finished.txt"):
            (directory/name).touch()
    path=f.path(directory)
    p=path.identity
    points=tuple(PreparedPoint(p.profile,p.task_id,p.stage,p.arm,p.plan_fingerprint,p.binary_sha256,p.raw_record,
        step,step-16,.4,.1,.6,.5,"") for step in (16,32))
    for name,frame in (("points.csv.gz",rows_frame(points)),("paths.csv",rows_frame((path,)))):
        write_table(preparation,name,frame)
        frame.to_csv(preparation/name,index=False)
    return preparation,path,points


def test_missing_disk_stays_pending_and_does_not_measure_or_fill_zero(tmp_path,monkeypatch):
    prep,_,_=_prepared(tmp_path,monkeypatch,available=False)
    monkeypatch.setattr(raw,"_measure",lambda *args:pytest.fail("missing media opened"))
    output=tmp_path/"collection"
    result=raw.collect_raw(prep,output)
    assert result.status=="pending" and result.pending_paths==result.planned_paths==1
    assert result.completed_paths==result.points==0 and result.paths[0].missing_files
    assert pd.read_csv(output/"indices.csv.gz").empty and pd.read_csv(output/"pathways.csv").empty
    assert result.preparation_sha256==sha(prep/"preparation.json")


def test_endpoint_policy_is_frozen_before_receipt_reuse(tmp_path,monkeypatch):
    prep,_,_=_prepared(tmp_path,monkeypatch,available=False)
    output=tmp_path/"collection"
    result=raw.collect_raw(prep,output)
    config=json.loads((output/"configuration.json").read_text())
    assert result.endpoint_complement is config["endpoint_complement"] is True
    assert config["metric_protocol"]["endpoint_complement"] is True
    with pytest.raises(ValueError,match="manifest changed"):
        raw.collect_raw(prep,output,endpoint_complement=False)


def test_measurement_uses_prepared_states_and_canonical_full_prefix(tmp_path,monkeypatch):
    import smp_bindings
    from ehk.metrics import trajectory
    _,path,points=_prepared(tmp_path,monkeypatch,available=True)
    calls=[]
    class SavedRecord:
        def __init__(self,base,metadata):
            assert metadata.UniqueName=="hold"
            self.directory=str(tmp_path/"disk"/path.raw_record)
            self.metadata=metadata
            self.max_step=32
            self.opinions=np.zeros((33,4))
        def get_graph(self,step):
            return nx.DiGraph()
        def __enter__(self): return self
        def __exit__(self,*args): pass
    monkeypatch.setattr(smp_bindings,"RawSimulationRecord",SavedRecord)
    monkeypatch.setattr(MeasurementEvaluator,"point",lambda self,record,step:(calls.append(step) or step/100,.2,.4))
    monkeypatch.setattr(trajectory,"sample_indices",lambda end,point:(np.array([0,8,32]),np.array([0,.5,1]),np.array([0,.2,.7]),np.zeros(3)))
    result=raw._measure(tmp_path/"disk"/path.raw_record,path,points)
    assert calls==[16,32] and [p.I_p for p in result.points]==[.16,.32]
    assert result.path.I_w==pytest.approx(.725) and result.path.endpoint_complement
    assert result.path.prefix_start==0 and result.path.trajectory_end==32
    assert result.path.sampling_method==core.IW_SAMPLING_METHOD
    integral=raw._measure(tmp_path/"disk"/path.raw_record,path,points,endpoint_complement=False)
    assert integral.path.I_w==pytest.approx(.425) and integral.path.endpoint_complement is False


def test_raw_metadata_mismatch_rejected_before_metrics(tmp_path,monkeypatch):
    _,path,points=_prepared(tmp_path,monkeypatch,available=True)
    with pytest.raises(ValueError,match="confidence radius differs"):
        raw._validate_raw(tmp_path/"disk"/path.raw_record,replace(path,epsilon=.8),points)


def test_collection_recovery_preserves_committed_generation_without_raw_disk(tmp_path,monkeypatch):
    prep,path,_=_prepared(tmp_path,monkeypatch,available=True)
    monkeypatch.setattr(raw,"_measure",lambda *args,**kwargs:f.measurement(path))
    output=tmp_path/"collection"
    first=raw.collect_raw(prep,output)
    before=sha(output/"collection.json")
    monkeypatch.setattr(raw,"_required_files",lambda _: ("unmounted",))
    monkeypatch.setattr(raw,"_measure",lambda *args,**kwargs:pytest.fail("committed metrics recomputed"))
    second=raw.collect_raw(prep,output)
    assert first==second and second.completed_paths==1
    assert sha(output/"collection.json")==before
    assert json.loads((output/"progress.json").read_text())["counts"]["cached"]==1


def test_arm_parameter_overrides_verified_against_preparation(tmp_path,monkeypatch):
    _,path,points=_prepared(tmp_path,monkeypatch,available=True)
    directory=tmp_path/"disk"/path.raw_record
    expected=replace(f.metadata(),PostRetainCount=0,RecsysCount=10,RecSysParams=RecommenderParams(Steepness=1,OpRandomNoiseStd=2))
    actual=replace(expected,RecSysParams=RecommenderParams(Steepness=4,OpRandomNoiseStd=0))
    atomic_json(directory/"metadata.json",actual)
    path=replace(path,metadata_json=canonical(expected),expected_rec_params_json=canonical(actual.RecSysParams))
    assert raw._validate_raw(directory,path,points)==actual
    with pytest.raises(ValueError,match="recommendation parameters"):
        raw._validate_raw(directory,replace(path,expected_rec_params_json=canonical(expected.RecSysParams)),points)


def test_changed_raw_sources_cannot_reuse_committed_values(tmp_path,monkeypatch):
    prep,path,_=_prepared(tmp_path,monkeypatch,available=True)
    monkeypatch.setattr(raw,"_measure",lambda *args,**kwargs:f.measurement(path))
    output=tmp_path/"collection"
    assert raw.collect_raw(prep,output).status=="complete"
    (tmp_path/"disk"/path.raw_record/"events.db").write_bytes(b"changed")
    with pytest.raises(ValueError,match="source files changed"):
        raw.collect_raw(prep,output)
