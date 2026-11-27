"""Numerical reporting preserves exact cohorts, full prefixes and paired effects."""

from dataclasses import replace
import inspect
import json
import subprocess

import numpy as np
import numpy.testing
import pandas as pd
import pytest

from common.io.json import atomic_json, sha
from experiments.paper.recommendation_shared.storage import write_table
from experiments.paper.kde_approximation.report import (
    benchmark_summary,
    load_inputs,
    paired_effects,
    point_errors,
    report,
    trajectory_errors,
)
from experiments.paper.metric_comparison.core import PATH_KEYS
from tests.experiments import numerical_fixtures as fixtures
from experiments.paper.kde_approximation.collector import _source_configuration
from experiments.paper.kde_approximation.contracts import Configuration, Selection, ReportIdentity, CollectionManifest
from experiments.paper.metric_comparison.contracts import PathKey
from ehk.metrics.protocol import metric_protocol
from experiments.paper.metric_comparison.report import paired_table


@pytest.fixture(autouse=True)
def forbid_process(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("reporting must not launch a native process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def _manifest(directory, name, value):
    for path in directory.glob("*.csv*"):
        write_table(directory, path.name, pd.read_csv(path, float_precision="round_trip"))
    atomic_json(directory / name, replace(value,files={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name != name}))


def _fixture(tmp_path):
    prep, baseline, collection = (tmp_path / name for name in ("preparation", "baseline", "collection"))
    for directory in (prep, baseline, collection):
        directory.mkdir()
    endpoints, prepared, points, paths = [], [], [], []
    for task in ("a", "b"):
        for arm in ("hold", "switch"):
            key = dict(profile="staged", task_id=task, stage="middle", arm=arm)
            changed = arm == "switch"
            consensus = changed and task == "a"
            final_p = 0 if consensus else .81 if changed else .8
            final_ip = .95 if consensus else .79 if changed else .8
            state = {16: (.2, .6, .5), 32: (final_ip, 1.0 if consensus else .98, 1.0 if consensus else .9)}
            endpoint = {**key, "start_step": 16, "end_step": 32, "followup_steps": 16,
                        "sensitivity": "core", "epsilon": .45, "history": 0,
                        "influence": .01, "rewiring": .1, "repost": .1,
                        "I_w": .2 if changed else .1, "endpoint_complement": False, "terminal_outcome": "consensus" if consensus else "two_groups"}
            for position, step in (("start", 16), ("end", 32)):
                ip, is_, ih = state[step]
                for metric, value in {"I_p": ip, "I_s": is_, "I_h": ih,
                                      "separation": .5 if position == "start" else final_p,
                                      "excess_concordance": .05 if position == "start" else 0 if consensus else .2,
                                      "edge_concordance": .55 if position == "start" else .5 if consensus else .7,
                                      "pair_concordance": .5}.items():
                    endpoint[f"{position}_{metric}"] = value
                prepared.append({**key, "step": step})
            endpoints.append(endpoint)
            for size in (1021, 2041):
                factor = 1021 / size
                paths.append({**key, "bins": size, "end_step": 32, "endpoint_complement": False,
                              "baseline_steps_json": "[0,16,32]", "candidate_steps_json": "[0,8,16,32]",
                              "baseline_Iw": endpoint["I_w"], "source_Iw": endpoint["I_w"], "fixed_Iw": endpoint["I_w"] + .001 * factor * (2 if changed else 1),
                              "adaptive_Iw": endpoint["I_w"] + .003 * factor * (2 if changed else 1),
                              "baseline_active_step": 24, "fixed_active_step": 25, "adaptive_active_step": 26})
                for grid, steps in (("prepared", (16, 32)), ("baseline_adaptive", (0, 16, 32)),
                                    ("candidate_adaptive", (0, 8, 16, 32))):
                    for step in steps:
                        exact = state.get(step, (.1, .3, .2)) if step != 8 else (np.nan, np.nan, np.nan)
                        current = state.get(step, (.1, .3, .2))
                        error = factor * (.002 if changed and step == 32 else .001 if step == 32 else .0001)
                        points.append({**key, "bins": size, "grid": grid, "step": step,
                                       **dict(zip(("exact_Ip", "exact_Is", "exact_Ih"), exact)),
                                       **dict(zip(("source_Ip", "source_Is", "source_Ih"), exact)),
                                       "approx_Ip": current[0] + error, "approx_Is": current[1] + error / 10,
                                       "approx_Ih": current[2]})
    pd.DataFrame(prepared).to_csv(prep / "points.csv.gz", index=False)
    _manifest(prep,"preparation.json",fixtures.prepared(points=8,paths=4))
    pd.DataFrame(endpoints).to_csv(baseline / "endpoint_states.csv.gz", index=False)
    _manifest(baseline,"report.json",fixtures.report(status="complete",paths=4,points=8,endpoint_complement=False,
        preparation_sha256=sha(prep / "preparation.json"),report_source_sha256=sha(inspect.getfile(paired_table))))
    benchmarks = pd.DataFrame([{"bins": size, "repeat": repeat, "same_input_sha": "same-state",
                                "exact_total_seconds": 2., "exact_evaluate_seconds": 1.,
                                "prepare_seconds": .1, "evaluate_seconds": .1 * size / 1021,
                                "binned_total_seconds": .1 + .1 * size / 1021}
                               for size in (1021, 2041) for repeat in (0, 1)])
    pd.DataFrame(points).to_csv(collection / "points.csv.gz", index=False)
    pd.DataFrame(paths).to_csv(collection / "paths.csv.gz", index=False)
    benchmarks.to_csv(collection / "benchmarks.csv", index=False)
    keys=tuple(PathKey("staged",task,"middle",arm) for task in ("a","b") for arm in ("hold","switch"))
    selection=Selection("representative",2,2,2,4,4,(("staged","a"),("staged","b")),
        {"staged":2},keys,(),(),"declared fixture tasks","declared adaptive stress states")
    config=Configuration(schema=1,preparation=str(prep),baseline=str(baseline),endpoint_complement=False,
        preparation_sha256=sha(prep / "preparation.json"),run_status_sha256="fixture-run",
        collection_sha256="fixture-collection",collection_configuration_fingerprint="fixture-configuration",
        comparison_sha256="fixture-comparison",baseline_report=ReportIdentity(str(baseline),sha(baseline / "report.json")),
        selection=selection,selected_paths=keys,bins=(1021,2041),benchmark_states=(),repeats=2,
        metrics=_source_configuration(),metric_protocol=metric_protocol(endpoint_complement=False),exact_reference="exact")
    atomic_json(collection / "configuration.json",config)
    _manifest(collection,"collection.json",CollectionManifest(schema=1,status="complete",planned_paths=4,
        completed_paths=4,endpoint_complement=False,pending_paths=0,failed_paths=0,selection=selection,
        paths=(),configuration_sha256=sha(collection / "configuration.json"),files={}))
    return prep, baseline, collection


def _rewrite_collection(collection, name, frame):
    write_table(collection, name, frame)
    frame.to_csv(collection / name, index=False)
    manifest = json.loads((collection / "collection.json").read_text())
    manifest["files"][name] = sha(collection / name)
    manifest["files"]["analysis.sqlite"] = sha(collection / "analysis.sqlite")
    atomic_json(collection / "collection.json", manifest)


def test_report_authenticates_counts_and_never_fills_unmeasured_candidate_states(tmp_path,monkeypatch):
    prep, baseline, collection = _fixture(tmp_path)
    destination = tmp_path / "report"
    result = report(prep, baseline, collection, destination, make_figures=False)
    assert result.selected_tasks == 2 and result.selected_paths == 4
    assert result.scope.mode == "representative"
    summary = pd.read_csv(destination / "state_errors.csv")
    candidate = summary[(summary["grid"] == "candidate_adaptive") & (summary.metric == "I_p")]
    assert candidate["unavailable_exact_states"].tolist() == [4, 4]
    assert candidate["n"].tolist() == [12, 12]
    assert all(sha(destination / name) == expected for name, expected in result.files.items())
    assert "没有据此宣布科学结论等价" in (destination / "report.md").read_text(encoding="utf-8")
    assert "不能外推整轮加速" in (destination / "report.md").read_text(encoding="utf-8")
    def forbidden(*args,**kwargs):
        pytest.fail("resuming the authenticated report started a new calculation")
    monkeypatch.setattr("experiments.paper.kde_approximation.report._produce_report",forbidden)
    assert report(prep,baseline,collection,destination,make_figures=False) == result


def test_fixed_and_adaptive_errors_and_step_changes_remain_separate(tmp_path):
    prep, baseline, collection = _fixture(tmp_path)
    _, paths, _, _, _ = load_inputs(prep, baseline, collection)
    detailed, summary = trajectory_errors(paths)
    first = detailed.iloc[0]
    assert first.fixed_Iw_error == pytest.approx(.001)
    assert first.adaptive_Iw_error == pytest.approx(.003)
    assert first.grid_Iw_change == pytest.approx(.002)
    assert first.added_step_count == 1 and first.removed_step_count == 0
    assert not first.same_accepted_steps
    assert set(summary.quantity) >= {"fixed_Iw_error", "adaptive_Iw_error", "grid_Iw_change", "adaptive_active_step_error"}


def test_paired_effects_reuse_same_cohort_and_consensus_subsets(tmp_path):
    prep, baseline, collection = _fixture(tmp_path)
    points, _, _, endpoints, _ = load_inputs(prep, baseline, collection)
    pairs, effects, baseline_pairs = paired_effects(points, endpoints)
    assert len(baseline_pairs) == 2
    assert (pairs[pairs.metric.isin(("I_h", "separation", "excess_concordance"))].delta_error == 0).all()
    selected = effects[(effects.bins == 1021) & (effects.metric == "I_p")].set_index("scope")
    assert selected.loc["middle_P_ge_0.4_E_lt_0.1", "mean_exact"] == pytest.approx(.07)
    assert selected.loc["middle_P_ge_0.4_E_lt_0.1", "mean_error"] == pytest.approx(.001)
    assert selected.loc["middle_consensus_gains", "n_exact"] == 1
    assert selected.loc["middle_remaining", "n_exact"] == 1
    assert pd.isna(selected.loc["middle_consensus_gains", "interval_side_changed"])


@pytest.mark.parametrize("fault,match", [("duplicate", "duplicate"), ("missing_state", "prepared observations"),
                                         ("prefix", "full 0..end"), ("homophily", "homophily"),
                                         ("horizon", "full 0..end"), ("unmatched_exact", "frozen endpoint")])
def test_rejects_corrupted_state_grid_and_identity(tmp_path, fault, match):
    prep, baseline, collection = _fixture(tmp_path)
    name = "paths.csv.gz" if fault in ("prefix", "horizon") else "points.csv.gz"
    frame = pd.read_csv(collection / name, float_precision="round_trip")
    if fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif fault == "missing_state":
        frame = frame.drop(index=0)
    elif fault == "prefix":
        frame.loc[0, "baseline_steps_json"] = "[16,32]"
    elif fault == "horizon":
        frame.loc[0, "end_step"] = 48
    elif fault == "homophily":
        frame.loc[0, "approx_Ih"] += .1
    elif fault == "unmatched_exact":
        frame.loc[0, "exact_Ip"] += .1
    _rewrite_collection(collection, name, frame)
    with pytest.raises(ValueError, match=match):
        load_inputs(prep, baseline, collection)


def test_rejects_incomplete_task_policies_even_if_both_grids_agree(tmp_path):
    prep, baseline, collection = _fixture(tmp_path)
    for name in ("points.csv.gz", "paths.csv.gz"):
        frame = pd.read_csv(collection / name)
        _rewrite_collection(collection, name, frame[frame.arm != "switch"])
    with pytest.raises(ValueError, match="omit baseline stages or policies"):
        load_inputs(prep, baseline, collection)


def test_hash_and_baseline_generation_mismatches_are_rejected(tmp_path):
    prep, baseline, collection = _fixture(tmp_path)
    with (collection / "benchmarks.csv").open("a") as stream:
        stream.write("corruption\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_inputs(prep, baseline, collection)


def test_frozen_baseline_cannot_be_relabelled_with_a_different_endpoint_policy(tmp_path):
    prep, baseline, collection = _fixture(tmp_path)
    manifest = json.loads((collection / "collection.json").read_text())
    manifest["endpoint_complement"] = True
    atomic_json(collection / "collection.json", manifest)
    with pytest.raises(ValueError, match="endpoint_complement policies disagree"):
        load_inputs(prep, baseline, collection)


def test_whole_missing_kde_grid_cannot_be_reported_as_complete(tmp_path):
    prep, baseline, collection = _fixture(tmp_path)
    for name in ("points.csv.gz", "paths.csv.gz", "benchmarks.csv"):
        frame = pd.read_csv(collection / name, float_precision="round_trip")
        _rewrite_collection(collection, name, frame[frame.bins != 2041])
    with pytest.raises(ValueError, match="frozen configuration"):
        load_inputs(prep, baseline, collection)


def test_complete_manifest_requires_complete_path_counts(tmp_path):
    prep, baseline, collection = _fixture(tmp_path)
    manifest = json.loads((collection / "collection.json").read_text())
    manifest["planned_paths"] = 5
    atomic_json(collection / "collection.json", manifest)
    with pytest.raises(ValueError, match="path coverage"):
        load_inputs(prep, baseline, collection)


def test_benchmark_separates_prepared_and_end_to_end_and_requires_matched_input(tmp_path):
    _, _, collection = _fixture(tmp_path)
    benchmark = pd.read_csv(collection / "benchmarks.csv")
    detailed, summary = benchmark_summary(benchmark)
    assert detailed.iloc[0].end_to_end_speedup == pytest.approx(10)
    assert detailed.iloc[0].prepared_evaluation_speedup == pytest.approx(10)
    assert set(summary.quantity) >= {"prepare_seconds", "evaluate_seconds", "end_to_end_speedup"}
    without_exact, _ = benchmark_summary(benchmark.drop(columns="exact_evaluate_seconds"))
    assert without_exact.prepared_evaluation_speedup.isna().all()
    benchmark.loc[0, "same_input_sha"] = "another-state"
    with pytest.raises(ValueError, match="matched prepared inputs"):
        benchmark_summary(benchmark)
    empty, summary = benchmark_summary(benchmark.iloc[:0])
    assert empty.empty and summary.empty


def test_errors_include_worst_case_without_relying_on_correlation(tmp_path):
    prep, baseline, collection = _fixture(tmp_path)
    points, *_ = load_inputs(prep, baseline, collection)
    _, summary, worst = point_errors(points)
    selected = summary[(summary.bins == 1021) & (summary.grid == "prepared") & (summary.metric == "I_p")].iloc[0]
    assert selected.maximum_absolute_error == pytest.approx(.002)
    assert selected.n == 8
    assert not worst.empty and set(PATH_KEYS) <= set(worst)
