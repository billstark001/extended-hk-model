"""Raw collection uses the existing indices and preserves missing-media status."""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from experiments.paper.metric_comparison import core, raw


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("metric preparation must not run a native process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def _prepared(tmp_path, monkeypatch, *, available):
    preparation = tmp_path / "preparation"
    preparation.mkdir()
    (preparation / "preparation.json").write_text('{"schema":1}')
    monkeypatch.setattr(core, "authenticate_preparation", lambda path: {"schema": 1})
    monkeypatch.setattr(raw, "_source_configuration", lambda: {"existing_metrics": "fixture"})
    directory = tmp_path / "disk" / "raw" / "task" / "late" / "hold"
    metadata = {"UniqueName": "hold", "NodeCount": 4, "HKParams": {"Tolerance": .45}, "RNG": {"Seed1": "1"}}
    identity = {"profile": "staged", "task_id": "task", "stage": "late", "arm": "hold",
                "plan_fingerprint": "plan", "binary_sha256": "binary", "raw_record": "raw/task/late/hold"}
    points = [{**identity, "step": step, "relative_step": step - 16, "separation": .4,
               "excess_concordance": .1, "edge_concordance": .6, "pair_concordance": .5}
              for step in (16, 32)]
    path = {**identity, "raw_directory": str(directory), "start_step": 16, "end_step": 32,
            "epsilon": .45, "metadata_json": json.dumps(metadata)}
    pd.DataFrame(points).to_csv(preparation / "points.csv.gz", index=False)
    pd.DataFrame([path]).to_csv(preparation / "paths.csv", index=False)
    if available:
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(json.dumps(metadata))
        (directory / "trajectory.json").write_text('{}')
        (directory / "events.db").touch()
        (directory / "graph-000000.msgpack").touch()
        (directory / "finished.txt").touch()
        (directory / "observations.json").write_text(json.dumps({
            "stopped_at": 32, "endpoints": [{"step": p["step"], "pathway": {
                name: p[name] for name in ("separation", "excess_concordance", "edge_concordance", "pair_concordance")
            }} for p in points],
        }))
    return preparation, path, pd.DataFrame(points)


def test_missing_disk_stays_pending_and_does_not_measure_or_fill_zero(tmp_path, monkeypatch):
    preparation, _, _ = _prepared(tmp_path, monkeypatch, available=False)
    monkeypatch.setattr(raw, "_measure", lambda *args: pytest.fail("missing media was opened"))
    output = tmp_path / "collected"
    report = raw.collect_raw(preparation, output)
    assert report["status"] == "pending"
    assert report["pending_paths"] == report["planned_paths"] == 1
    assert report["completed_paths"] == report["points"] == 0
    assert pd.read_csv(output / "legacy_points.csv.gz").empty
    assert pd.read_csv(output / "legacy_paths.csv").empty
    assert report["paths"][0]["missing_files"]
    assert report["preparation_sha256"] == core.sha(preparation / "preparation.json")


def test_endpoint_policy_is_frozen_before_receipt_reuse(tmp_path, monkeypatch):
    preparation, _, _ = _prepared(tmp_path, monkeypatch, available=False)
    output = tmp_path / "collected"
    result = raw.collect_raw(preparation, output)
    configuration = json.loads((output / "configuration.json").read_text())
    assert result["endpoint_complement"] is configuration["endpoint_complement"] is True
    assert configuration["metric_protocol"]["endpoint_complement"] is True
    with pytest.raises(ValueError, match="manifest changed"):
        raw.collect_raw(preparation, output, endpoint_complement=False)


def test_measurement_reuses_existing_exact_states_and_full_prefix_iw(tmp_path, monkeypatch):
    import smp_bindings

    from ehk.micro import observables

    _, path, points = _prepared(tmp_path, monkeypatch, available=True)
    calls = []

    class SavedRecord:
        def __init__(self, base, metadata):
            assert metadata["UniqueName"] == "hold"
            self.max_step = 32

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(smp_bindings, "RawSimulationRecord", SavedRecord)
    monkeypatch.setattr(observables, "calc_distance_components", lambda record, step: (calls.append(step) or step / 100, .2, 0., 0.))
    monkeypatch.setattr(observables, "calc_homophily", lambda record, step: .4)
    from ehk.metrics import trajectory

    monkeypatch.setattr(trajectory, "sample_indices", lambda end, point: (
        np.array([0, 8, 32]), np.array([0, .5, 1]), np.array([0, .2, .7]), np.zeros(3),
    ))
    measured = raw._measure(tmp_path / "disk" / path["raw_record"], path, points)
    assert calls == [16, 32]
    assert [p["I_p"] for p in measured["points"]] == [.16, .32]
    assert measured["path"]["I_w"] == pytest.approx(.725)
    assert measured["path"]["endpoint_complement"] is True
    integral = raw._measure(tmp_path / "disk" / path["raw_record"], path, points, endpoint_complement=False)
    assert integral["path"]["I_w"] == pytest.approx(.425)
    assert integral["path"]["endpoint_complement"] is False
    assert measured["path"]["prefix_start"] == 0
    assert measured["path"]["trajectory_end"] == 32
    assert measured["path"]["sampling_method"] == core.IW_SAMPLING_METHOD


def test_raw_mismatch_is_rejected_before_computing_metrics(tmp_path, monkeypatch):
    _, path, points = _prepared(tmp_path, monkeypatch, available=True)
    points.loc[0, "separation"] = .8
    with pytest.raises(ValueError, match="raw separation differs"):
        raw._validate_raw(tmp_path / "disk" / path["raw_record"], path, points)


def test_collection_resumes_completed_paths_without_raw_disk(tmp_path, monkeypatch):
    preparation, path, _ = _prepared(tmp_path, monkeypatch, available=True)
    measured = {"points": [{**{key: path[key] for key in (*raw.PATH_KEYS, *raw.SOURCE_KEYS)},
                            "step": step, "I_p": .2, "I_s": .3, "I_h": .4} for step in (16, 32)],
                "path": {**{key: path[key] for key in (*raw.PATH_KEYS, *raw.SOURCE_KEYS)},
                         "I_w": .5, "sampling_method": core.IW_SAMPLING_METHOD, "prefix_start": 0, "trajectory_end": 32,
                         "endpoint_complement": True}}
    monkeypatch.setattr(raw, "_measure", lambda *args, **kwargs: measured)
    output = tmp_path / "collected"
    assert raw.collect_raw(preparation, output)["status"] == "complete"
    monkeypatch.setattr(raw, "_required_files", lambda path: ["unmounted"])
    monkeypatch.setattr(raw, "_measure", lambda *args: pytest.fail("completed metrics were recomputed"))
    result = raw.collect_raw(preparation, output)
    assert result["status"] == "complete"
    assert result["paths"][0]["status"] == "cached"
    assert result["completed_paths"] == 1


def test_arm_parameter_overrides_are_verified_against_the_prepared_configuration(tmp_path, monkeypatch):
    _, path, points = _prepared(tmp_path, monkeypatch, available=True)
    directory = tmp_path / "disk" / path["raw_record"]
    metadata = json.loads((directory / "metadata.json").read_text())
    default_parameters = {"Steepness": 1, "OpRandomNoiseStd": 2}
    expected = {**metadata, "PostRetainCount": 0, "RecsysCount": 10,
                "RecSysParams": default_parameters}
    metadata.update(PostRetainCount=0, RecsysCount=10,
                    RecSysParams={"Steepness": 4, "OpRandomNoiseStd": 0})
    (directory / "metadata.json").write_text(json.dumps(metadata))
    path["metadata_json"] = json.dumps(expected)
    path["expected_rec_params_json"] = json.dumps(metadata["RecSysParams"])
    assert raw._validate_raw(directory, path, points) == metadata
    path["expected_rec_params_json"] = json.dumps(default_parameters)
    with pytest.raises(ValueError, match="recommendation parameters"):
        raw._validate_raw(directory, path, points)


def test_changed_raw_sources_cannot_reuse_cached_metric_values(tmp_path, monkeypatch):
    preparation, path, _ = _prepared(tmp_path, monkeypatch, available=True)
    measured = {"points": [], "path": {
        **{key: path[key] for key in (*raw.PATH_KEYS, *raw.SOURCE_KEYS)},
        "I_w": .5, "sampling_method": core.IW_SAMPLING_METHOD,
        "prefix_start": 0, "trajectory_end": 32, "endpoint_complement": True,
    }}
    monkeypatch.setattr(raw, "_measure", lambda *args, **kwargs: measured)
    output = tmp_path / "collected"
    assert raw.collect_raw(preparation, output)["status"] == "complete"
    (tmp_path / "disk" / path["raw_record"] / "events.db").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source files changed"):
        raw.collect_raw(preparation, output)
