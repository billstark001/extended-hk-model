"""Experimental collection preserves frozen paths and separates three numerical grids."""

import json
import subprocess
from pathlib import Path
from dataclasses import dataclass,replace
from ehk.domain import rows_frame,to_wire
from tests.experiments import numerical_fixtures as f
from experiments.paper.metric_comparison.contracts import (CollectionReceipt,EvaluationRecord,PipelineStatus,PipelinePaths,
    AlignedEvidence,CollectionConfiguration,PathKey)
from experiments.paper.kde_approximation.contracts import (BaselineEvidence,ReportIdentity,BaselinePoint,RunStatus,RunPaths)
from experiments.paper.kde_approximation.kernel import KernelDiagnostics

import networkx as nx
import numpy as np
import numpy.testing
import pandas as pd
import pytest

from common.io.json import atomic_json, fingerprint, sha
from common.storage.documents import DocumentStore
from ehk.metrics import trajectory
from ehk.micro import observables
from experiments.paper.kde_approximation import __main__ as cli
from experiments.paper.kde_approximation import collector


@pytest.fixture(autouse=True)
def no_native_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("collector tests must not run model or probe processes")

    monkeypatch.setattr(subprocess, "Popen", forbidden)


@dataclass
class FakePrepared:
    step:int
    epsilon:float
    objective:int
    edge:int


class FakeCalculator:
    exact_calls = 0

    def __init__(self):
        self.reference = self

    def parameter_key(self):
        return ("fixture",)

    def prepare(self, graph, opinions, epsilon, step):
        return self.prepare_state(None, epsilon, step)

    def prepare_state(self, state, epsilon, step):
        return FakePrepared(step,epsilon,step,step)

    def prepare_opinion(self, opinions, epsilon, *, random_state):
        return random_state

    def prepare_edge(self, state, *, random_state):
        return random_state

    def opinion_index(self, step, *, method, bins=None, diagnostics=None):
        if diagnostics is not None:
            diagnostics.negative_count = 0
        return .1 + step / 100 + (0 if method == "exact" else .001 / bins)

    def edge_index(self, step, *, method, bins=None, diagnostics=None):
        if diagnostics is not None:
            diagnostics.negative_count = 0
        return .5 + step / 100 + (0 if method == "exact" else .002 / bins)

    def _diagnostics(self,prepared,*,method,bins=None,opinion_detail=None,edge_detail=None):
        return KernelDiagnostics(method=method,bins=bins,stride=None,opinion_samples=6,edge_samples=4,
            opinion_bandwidth=.01,edge_bandwidth=.01,opinion_constant=False,edge_constant=False)

    def exact(self, prepared):
        type(self).exact_calls += 1
        return self.opinion_index(prepared.step, method="exact"), self.edge_index(prepared.step, method="exact"), self._diagnostics(prepared,method="exact")

    def evaluate(self, prepared, bins):
        return (self.opinion_index(prepared.step, method="binned", bins=bins),
                self.edge_index(prepared.step, method="binned", bins=bins),
                self._diagnostics(prepared, method="linear_binned_gaussian", bins=bins))


def _fake_indices(end, point):
    steps = np.array([0, 4, 8] if point(0)[0] == .1 else [0, 2, 8])
    values = np.array([point(int(step)) for step in steps])
    return steps, values[:, 2], values[:, 0], values[:, 1]


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    import smp_bindings

    preparation, baseline = tmp_path / "preparation", tmp_path / "baseline"
    paths, points = [], []
    for profile, task, stage in (("staged", "one", "middle"), ("staged_timing", "two", "full")):
        for arm in ("hold", "switch"):
            raw_record = f"raw/{task}/{stage}/{arm}"
            directory = tmp_path / "disk" / raw_record
            directory.mkdir(parents=True)
            (directory / "input").write_bytes(b"frozen raw state")
            path=f.path(directory,profile=profile,task_id=task,stage=stage,arm=arm,start=0,end=8)
            paths.append(path)
            measured=f.measurement(path)
            measured=replace(measured,path=replace(measured.path,I_w=.18),evaluation=EvaluationRecord((0,4,8),True))
            with DocumentStore(baseline / "collection/analysis.sqlite") as store:
                store.put("metric_comparison.paths",fingerprint(path.identity),CollectionReceipt(
                    "baseline-configuration",path.identity,True,{"input":sha(directory/"input")},measured))
            for step in (0,8):
                points.append(BaselinePoint(path.profile,path.task_id,path.stage,path.arm,path.plan_fingerprint,
                    path.binary_sha256,path.raw_record,step,step,.5,.1,.6,.5,"",.1+step/100,.5+step/100,.2))
    frames = rows_frame(points), rows_frame(paths)
    monkeypatch.setattr(collector,"_load_baseline",lambda *args:(*frames,BaselineEvidence(True,"prep","run","collect",
        "baseline-configuration","compare",ReportIdentity(str(baseline/"report"),"report-sha"))))
    monkeypatch.setattr(collector, "_calculator", FakeCalculator)
    monkeypatch.setattr(collector.raw, "_required_files", lambda path: [] if (path / "input").is_file() else ["input"])
    monkeypatch.setattr(collector.raw, "_raw_sources", lambda path: {"input": sha(path / "input")})
    monkeypatch.setattr(collector.raw,"_validate_raw",lambda directory,path,points:f.metadata(directory.name))

    class Record:
        def __init__(self, base, metadata):
            self.directory = str(Path(base) / metadata.UniqueName)
            self.metadata, self.max_step = metadata, 8
            self.opinions = np.tile(np.array([-.8, -.2, .1, .7]), (9, 1))
            self.graph = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0)])

        def get_graph(self, step):
            return self.graph

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(smp_bindings, "RawSimulationRecord", Record)
    from ehk.metrics import homophily
    monkeypatch.setattr(homophily, "normalize_homophily", lambda *args: .2)
    collector._WORKER_CONTEXT = None
    monkeypatch.setattr(trajectory, "sample_indices", _fake_indices)
    return preparation, baseline, frames


def test_three_grids_keep_baseline_curve_and_activity_cost_separate(tmp_path, evidence):
    preparation, baseline, _ = evidence
    output = tmp_path / "experiment"
    result = collector.collect(preparation, baseline, output, workers=1, benchmark_states=2, repeats=1)
    assert result.status == "complete" and result.completed_paths == 4
    points = pd.read_csv(output / "points.csv.gz")
    paths = pd.read_csv(output / "paths.csv.gz")
    assert set(points["grid"]) == {"prepared", "baseline_adaptive", "candidate_adaptive"}
    assert len(paths) == 4 * len(collector.DEFAULT_BINS)
    for row in paths.to_dict("records"):
        assert json.loads(row["baseline_steps_json"]) == [0, 4, 8]
        assert json.loads(row["candidate_steps_json"]) == [0, 2, 8]
        assert row["baseline_Iw"] == pytest.approx(.18)
        assert row["endpoint_complement"] is True
        assert row["fixed_Iw"] == pytest.approx(row["adaptive_Iw"])
        assert row["extra_exact_calls"] == 1
        assert row["extra_exact_seconds"] >= 0
        assert row["baseline_active_step"] == row["fixed_active_step"] == row["adaptive_active_step"] == 8
    off_grid = points[(points["grid"] == "candidate_adaptive") & (points["step"] == 2)]
    assert off_grid["exact_Ip"].isna().all() and off_grid["exact_Is"].isna().all()
    assert (points["approx_Ih"] == .2).all() and (points["exact_Ih"] == .2).all()
    # Identical cross-branch states still retain their own point diagnostics.
    assert points["opinion_bandwidth"].notna().all()
    config = json.loads((output / "configuration.json").read_text())
    assert config["selection"]["profile_tasks"] == {"staged": 1, "staged_timing": 1}
    assert result.configuration_sha256 == sha(output / "configuration.json")


def test_benchmark_repetitions_pair_exact_totals_and_deduplicate_actual_inputs(tmp_path, evidence):
    preparation, baseline, _ = evidence
    output = tmp_path / "experiment"
    collector.collect(preparation, baseline, output, workers=1, benchmark_states=8, repeats=3)
    bench = pd.read_csv(output / "benchmarks.csv")
    assert not bench.duplicated(["same_input_sha", "repeat", "bins"]).any()
    assert set(bench["repeat"]) == {0, 1, 2}
    assert set(bench["timing_order"]) == {"exact_first", "binned_first"}
    np.testing.assert_allclose(bench["exact_total_seconds"], bench["exact_prepare_seconds"] + bench["exact_evaluate_seconds"])
    np.testing.assert_allclose(bench["binned_total_seconds"], bench["prepare_seconds"] + bench["evaluate_seconds"])
    assert bench.groupby(["same_input_sha", "repeat"])["bins"].nunique().eq(4).all()


def test_receipt_recovery_does_not_repeat_computation(tmp_path, evidence, monkeypatch):
    preparation, baseline, _ = evidence
    output = tmp_path / "experiment"
    options = {"workers": 1, "benchmark_states": 2, "repeats": 1}
    collector.collect(preparation, baseline, output, **options)
    with DocumentStore(output / "analysis.sqlite", readonly=True) as store:
        original = list(store.values("kde_approximation.paths"))
    monkeypatch.setattr(collector, "_measure_path", lambda *args: pytest.fail("completed path was recalculated"))
    result = collector.collect(preparation, baseline, output, **options)
    assert result.status == "complete"
    assert all(row.status == "complete" for row in result.paths)
    with DocumentStore(output / "analysis.sqlite", readonly=True) as store:
        assert list(store.values("kde_approximation.paths")) == original


def test_baseline_policy_must_match_every_approximation_curve(tmp_path, evidence):
    preparation, baseline, _ = evidence
    with pytest.raises(ValueError, match="baseline endpoint_complement differs"):
        collector.collect(preparation, baseline, tmp_path / "mismatch", endpoint_complement=False)


def test_explicit_integral_only_baseline_keeps_fixed_and_adaptive_consistent(tmp_path, evidence, monkeypatch):
    preparation, baseline, _ = evidence
    points, paths, provenance = collector._load_baseline(preparation, baseline)
    provenance=replace(provenance,endpoint_complement=False)
    monkeypatch.setattr(collector, "_load_baseline", lambda *args: (points, paths, provenance))
    with DocumentStore(baseline / "collection/analysis.sqlite") as store:
        for saved in list(store.values("metric_comparison.paths")):
            saved["result"]["path"].update(I_w=.016, endpoint_complement=False)
            store.put("metric_comparison.paths", fingerprint(saved["identity"]), saved)
    output = tmp_path / "integral-only"
    collected = collector.collect(preparation, baseline, output, workers=1, benchmark_states=2, repeats=1,
                                  endpoint_complement=False)
    assert collected.status == "complete" and collected.endpoint_complement is False
    rows = pd.read_csv(output / "paths.csv.gz")
    np.testing.assert_allclose(rows["baseline_Iw"], .016)
    np.testing.assert_allclose(rows["fixed_Iw"], rows["adaptive_Iw"], atol=1e-14)
    assert not rows["endpoint_complement"].any()


def test_changed_configuration_and_raw_inputs_cannot_reuse_receipts(tmp_path, evidence):
    preparation, baseline, frames = evidence
    output = tmp_path / "experiment"
    collector.collect(preparation, baseline, output, workers=1, benchmark_states=2, repeats=1)
    with pytest.raises(ValueError, match="manifest changed"):
        collector.collect(preparation, baseline, output, workers=1, benchmark_states=2, repeats=2)
    raw_directory = Path(frames[1].iloc[0]["raw_directory"])
    (raw_directory / "input").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen formal baseline"):
        collector.collect(preparation, baseline, output, workers=1, benchmark_states=2, repeats=1)


def test_representative_selection_is_deterministic_and_keeps_complete_groups(evidence):
    _, _, (points, paths) = evidence
    first, selection = collector.select_tasks(paths, points, max_tasks=1)
    shuffled, again = collector.select_tasks(paths.sample(frac=1, random_state=3), points, max_tasks=1)
    assert selection.task_keys == again.task_keys
    chosen = selection.task_keys[0]
    expected = paths[(paths["profile"] == chosen[0]) & (paths["task_id"] == chosen[1])]
    assert set(first["arm"]) == set(shuffled["arm"]) == set(expected["arm"])
    full, full_selection = collector.select_tasks(paths, points, max_tasks=1, full=True)
    assert len(full) == len(paths) and full_selection.mode == "full"


def test_representative_setting_coverage_is_scoped_to_each_profile(evidence):
    _, _, (points, paths) = evidence
    expanded_paths, expanded_points = [], []
    for setting in range(9):
        for profile in ("staged", "staged_timing"):
            task = f"{profile}-{setting}"
            path = paths[paths["profile"] == profile].copy()
            path["task_id"], path["influence"] = task, .01 * (setting + 1)
            point = points[points["profile"] == profile].copy()
            point["task_id"] = task
            expanded_paths.append(path)
            expanded_points.append(point)
    selected, selection = collector.select_tasks(pd.concat(expanded_paths), pd.concat(expanded_points), max_tasks=18)
    assert selection.profile_tasks == {"staged": 9, "staged_timing": 9}
    assert selected.groupby("profile")["influence"].nunique().to_dict() == {"staged": 9, "staged_timing": 9}


def test_process_dispatch_groups_all_paths_of_each_initial_state(tmp_path, evidence, monkeypatch):
    preparation, baseline, _ = evidence
    jobs = []

    def execute(function, tasks, workers, *, executor, on_result,event):
        assert workers == 2 and executor == "process"
        for job in tasks:
            jobs.append(job)
            on_result(job, function(job))

    monkeypatch.setattr(collector, "run_tasks", execute)
    result = collector.collect(preparation, baseline, tmp_path / "experiment", workers=2, benchmark_states=2, repeats=1)
    assert result.status == "complete" and len(jobs) == 4
    assert all(len(job.entries) == 1 for job in jobs)


def test_interrupt_commits_finished_paths_and_resumes_only_remaining_work(tmp_path, evidence, monkeypatch):
    preparation, baseline, _ = evidence
    output = tmp_path / "experiment"
    original = collector._measure_path
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(args[0].path.arm)
        if len(calls) == 2:
            raise KeyboardInterrupt("stop after one receipt")
        return original(*args, **kwargs)

    monkeypatch.setattr(collector, "_measure_path", interrupted)
    with pytest.raises(KeyboardInterrupt, match="one receipt"):
        collector.collect(preparation, baseline, output, workers=1, benchmark_states=2, repeats=1)
    with DocumentStore(output / "analysis.sqlite", readonly=True) as store:
        assert len(list(store.values("kde_approximation.paths"))) == 1
    assert json.loads((output / "collection.json").read_text())["status"] == "interrupted"
    monkeypatch.setattr(collector, "_measure_path", original)
    assert collector.collect(preparation, baseline, output, workers=1, benchmark_states=2, repeats=1).status == "complete"


def test_current_sources_must_still_match_the_baseline(tmp_path,monkeypatch):
    preparation,baseline=tmp_path/"preparation",tmp_path/"baseline"
    prep=f.prepared()
    atomic_json(preparation/"preparation.json",prep)
    monkeypatch.setattr(collector.baseline_reporting,"load_aligned",lambda *args:(None,None,AlignedEvidence(prep,f.comparison())))
    monkeypatch.setattr(collector.core,"_authenticate_files",lambda *args:None)
    status=PipelineStatus(schema=1,state="complete",phase="report",attempt=1,workers=1,attempts=1,
        make_figures=False,endpoint_complement=True,runner_sha256="run",paths=PipelinePaths("p","o","c","a","r","g"))
    atomic_json(baseline/"run_status.json",status)
    atomic_json(baseline/"collection/collection.json",f.collection())
    atomic_json(baseline/"report/report.json",f.report())
    sources=collector.raw._source_configuration()
    atomic_json(baseline/"collection/configuration.json",CollectionConfiguration(1,fingerprint(prep),sources,{},True,
        collector.raw.metric_protocol()))
    monkeypatch.setattr(collector.raw,"_source_configuration",lambda:replace(sources,collector_sha256="changed"))
    with pytest.raises(ValueError,match="sources differ"):
        collector._load_baseline(preparation,baseline)


def test_missing_raw_is_pending_and_never_zero_filled(tmp_path, evidence):
    preparation, baseline, frames = evidence
    Path(frames[1].iloc[0]["raw_directory"], "input").unlink()
    output = tmp_path / "experiment"
    result = collector.collect(preparation, baseline, output, workers=1, benchmark_states=2, repeats=1)
    assert result.status == "pending" and result.pending_paths == 1
    assert pd.read_csv(output / "paths.csv.gz")["task_id"].count() == 3 * 4


def test_cli_automatically_reports_complete_collection_and_stops_partial(tmp_path, monkeypatch):
    preparation, baseline, output = (tmp_path / name for name in ("prepared", "baseline", "experiment"))
    calls = []
    monkeypatch.setattr(cli.collector, "collect", lambda *args, **kwargs: (calls.append((args, kwargs)) or
                        KdeCollectionFixture("complete")))
    monkeypatch.setattr(cli.report, "report", lambda *args, **kwargs: (calls.append((args, kwargs)) or KdeReportFixture()))
    assert cli.run(preparation, baseline, output, workers=1, make_figures=False).state == "complete"
    assert calls[0][0] == (preparation, baseline, output / "collection")
    assert calls[1] == ((preparation, baseline / "report", output / "collection", output / "report"), {"make_figures": False})
    calls.clear()
    monkeypatch.setattr(cli.collector, "collect", lambda *args, **kwargs: KdeCollectionFixture("pending"))
    assert cli.run(preparation, baseline, tmp_path / "partial").state == "pending"
    assert calls == []


@pytest.mark.parametrize("target", ["baseline", "preparation", "completed"])
def test_cli_refuses_overlapping_or_completed_outputs_without_rewriting_status(tmp_path, monkeypatch, target):
    preparation, baseline, complete = (tmp_path / name for name in ("preparation", "baseline", "completed"))
    for path in (preparation, baseline, complete):
        path.mkdir()
        atomic_json(path / "run_status.json", {"state": "complete", "preserve": True})
    (complete / "report").mkdir()
    output = {"baseline": baseline, "preparation": preparation, "completed": complete}[target]
    before = sha(output / "run_status.json")
    monkeypatch.setattr(cli.collector, "collect", lambda *args, **kwargs: pytest.fail("invalid output started collection"))
    with pytest.raises(ValueError):
        cli.run(preparation, baseline, output)
    assert sha(output / "run_status.json") == before


def test_cli_options_are_routed_to_the_new_experiment(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or RunStatus(1,"pending","collect","run",True,RunPaths("p","b","c","r")))
    args = ["--preparation", str(tmp_path / "prepared"), "--baseline", str(tmp_path / "baseline"),
            "--output", str(tmp_path / "experiment"), "--workers", "1", "--max-tasks", "1",
            "--bins", "1021", "2041", "--benchmark-states", "2", "--repeats", "1", "--no-figures"]
    assert cli.main(args) == 1
    assert calls[0][1]["bins"] == [1021, 2041] and calls[0][1]["max_tasks"] == 1
    assert calls[0][1]["make_figures"] is False
    assert json.loads(capsys.readouterr().out)["state"] == "pending"


def KdeCollectionFixture(status):
    from experiments.paper.kde_approximation.contracts import CollectionManifest,Selection
    selection=Selection("representative",1,1,1,1,1,(("staged","one"),),{"staged":1},
        (PathKey("staged","one","late","hold"),),(),(),"fixture","fixture")
    return CollectionManifest(schema=1,files={},status=status,planned_paths=1,completed_paths=int(status=="complete"),
        endpoint_complement=True,pending_paths=int(status!="complete"),failed_paths=0,selection=selection,
        paths=(),configuration_sha256="config")


def KdeReportFixture():
    from experiments.paper.kde_approximation.contracts import Report
    return Report(schema=1,files={},status="complete",scope=KdeCollectionFixture("complete").selection,
        endpoint_complement=True,selected_tasks=1,selected_paths=1,baseline_paths=1,baseline_points=2,bins=(1021,),
        measured_rows=6,preparation_sha256="p",baseline_report_sha256="b",collection_sha256="c",configuration_sha256="f",
        report_source_sha256="r",paired_analysis_source_sha256="a")


def test_completed_paired_timings_survive_failure_before_path_commit(tmp_path,evidence,monkeypatch):
    preparation,baseline,_=evidence
    output=tmp_path/"recover-timing"
    original_measure,original_timing=collector._measure_path,collector._timing
    timing_calls=[]
    def measured(*args,**kwargs):
        timing_calls.append(args[1])
        return original_timing(*args,**kwargs)
    def interrupted(*args,**kwargs):
        original_measure(*args,**kwargs)
        raise KeyboardInterrupt("after timing, before receipt commit")
    monkeypatch.setattr(collector,"_timing",measured)
    monkeypatch.setattr(collector,"_measure_path",interrupted)
    with pytest.raises(KeyboardInterrupt,match="before receipt"):
        collector.collect(preparation,baseline,output,workers=1,benchmark_states=2,repeats=1)
    before=len(timing_calls)
    assert before>0
    with DocumentStore(output/"analysis.sqlite",readonly=True) as store:
        assert list(store.values("kde_approximation.paths"))==[]
    monkeypatch.setattr(collector,"_measure_path",original_measure)
    result=collector.collect(preparation,baseline,output,workers=1,benchmark_states=2,repeats=1)
    assert result.status=="complete" and len(timing_calls)==before
    timings=pd.read_csv(output/"benchmarks.csv")
    assert not timings.empty and set(timings.bins)==set(collector.DEFAULT_BINS)
    assert not timings.duplicated(["same_input_sha","repeat","bins"]).any()
