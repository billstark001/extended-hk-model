"""Authenticated alignment preserves unavailable metrics and exact paired identities."""

import json
import shutil
import subprocess

import pandas as pd
import pytest

from ehk.domain import freeze_plan, Binding, StagedResult, StageResult
from dataclasses import replace
from tests.experiments import recommendation_fixtures as fixtures
from experiments.paper.metric_comparison.contracts import CollectionManifest, CacheCounts, Coverage
from common.io.json import atomic_json, fingerprint, save_result, sha
from experiments.paper.metric_comparison.core import (
    IW_SAMPLING_METHOD,
    PATH_METRICS,
    POINT_METRICS,
    PATH_KEYS,
    POINT_KEYS,
    PROVENANCE,
    authenticate_preparation,
    compare,
    prepare,
)
from experiments.paper.recommendation.design import generate_tasks
from experiments.paper.recommendation_shared.storage import StudyStore


@pytest.fixture(autouse=True)
def forbid_native(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("metric preparation must not start native processes")
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def _bundle(tmp_path):
    root = tmp_path / "bundle"
    data = root / "data"
    task = generate_tasks("staged")[0]
    plan = freeze_plan(data / "study/plan.json", fixtures.plan(tasks=[task]))
    def point(step, start=0):
        return fixtures.point(step,start,separation=.2,excess=.1)
    def arm(stage,name,steps):
        return replace(fixtures.arm(endpoints=[point(s,steps[0]) for s in steps], k_major=1,
                raw_record=f"raw/{task.id}/{stage}/{name}"), stop_reason="horizon")
    record = StagedResult(reference=arm("reference","reference",[0,16]),
        stages={"middle":StageResult(checkpoint=point(16),arms={"switch":arm("middle","switch",[16,32])})})
    with StudyStore(data / "study") as store:
        store.put_task(task,record,plan=plan)
    atomic_json(data / "study/runtime/binding.json",Binding(binary_sha256="original-binary"))
    manifest=replace(fixtures.bundle_manifest(completed=1,planned=1,plan_fingerprint=plan.plan_fingerprint),
        source_dataset=str(tmp_path / "unmounted"), observation_dataset=str(data / "study"),
        files={p.relative_to(data).as_posix():sha(p) for p in data.rglob("*") if p.is_file()})
    atomic_json(data / "manifest.json",manifest)
    return root


def test_prepare_exports_reference_and_branch_with_pending_raw(tmp_path):
    bundle = _bundle(tmp_path)
    output = tmp_path / "prepared"
    before = {p: sha(p) for p in bundle.rglob("*") if p.is_file()}
    report = prepare([bundle], output)
    assert (report.points, report.paths, report.pending_raw_paths) == (4, 2, 2)
    assert report.missing_metrics == ("I_p", "I_s", "I_h", "I_w")
    points = pd.read_csv(output / "points.csv.gz")
    paths = pd.read_csv(output / "paths.csv")
    assert not points.duplicated(POINT_KEYS).any()
    assert set(points["stage"]) == {"reference", "middle"}
    assert points["excess_concordance"].tolist() == [.1] * 4
    assert paths["full_prefix_required"].all()
    assert not paths["available"].any()
    assert paths.loc[paths["stage"] == "middle", "start_step"].item() == 16
    assert json.loads(paths.iloc[0]["metadata_json"])["RNG"] == json.loads(paths.iloc[0]["seed_json"])
    assert authenticate_preparation(output).points == 4
    assert all(sha(p) == digest for p, digest in before.items())


def test_prepare_rejects_tampered_manifest_and_duplicate_generations(tmp_path):
    bundle = _bundle(tmp_path)
    with pytest.raises(ValueError, match="duplicate task"):
        prepare([bundle, bundle], tmp_path / "duplicates")
    binding = bundle / "data/study/runtime/binding.json"
    binding.write_text('{}')
    with pytest.raises(ValueError, match="study/runtime/binding.json"):
        prepare([bundle], tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def _collection(tmp_path, preparation, *, complement=True):
    from experiments.paper.recommendation_shared.storage import read_table, write_table
    root = tmp_path / "collection"
    root.mkdir()
    points = read_table(preparation, "points.csv.gz").iloc[:1][[*POINT_KEYS, *PROVENANCE]].copy()
    for name, value in zip(POINT_METRICS, [.3, .4, .5]):
        points[name] = value
    paths = read_table(preparation, "paths.csv").iloc[:1].copy()
    paths["I_w"], paths["sampling_method"], paths["prefix_start"] = .7, IW_SAMPLING_METHOD, 0
    paths["trajectory_end"], paths["endpoint_complement"] = paths["end_step"], complement
    paths = paths[[*PATH_KEYS, *PROVENANCE, *PATH_METRICS]]
    for name, frame in (("indices.csv.gz", points), ("pathways.csv", paths)):
        write_table(root, name, frame)
    atomic_json(root / "collection.json",CollectionManifest(schema=1,status="pending_metrics",
        endpoint_complement=complement,preparation_sha256=sha(preparation / "preparation.json"),
        preparation_fingerprint="fixture", planned_paths=2,selected_paths=1,completed_paths=1,
        pending_paths=1,failed_paths=0,points=len(points),paths=(),cache=CacheCounts(),
        files={"analysis.sqlite":sha(root / "analysis.sqlite")}))
    return root


def test_exact_sqlite_alignment_leaves_missing_measurements_pending(tmp_path):
    prepared = tmp_path / "prepared"
    prepare([_bundle(tmp_path)], prepared)
    collection = _collection(tmp_path, prepared)
    result = compare(prepared, collection, tmp_path / "compared")
    assert result.coverage == {"points": Coverage(4,1,3),"paths":Coverage(2,1,1)}
    from experiments.paper.recommendation_shared.storage import read_table
    aligned = read_table(tmp_path / "compared", "points_aligned.csv.gz")
    assert aligned["I_p"].isna().sum() == 3
    assert aligned.loc[aligned.metric_coverage == "both", "I_h"].item() == .5


@pytest.mark.parametrize("fault", ["duplicate", "provenance", "nonfinite", "foreign"])
def test_sqlite_alignment_rejects_ambiguous_or_unbound_measurements(tmp_path, fault):
    from experiments.paper.recommendation_shared.storage import read_table, write_table
    prepared = tmp_path / "prepared"
    prepare([_bundle(tmp_path)], prepared)
    collection = _collection(tmp_path, prepared)
    frame = read_table(collection, "indices.csv.gz")
    if fault == "duplicate": frame = pd.concat([frame, frame], ignore_index=True)
    elif fault == "provenance": frame.loc[0, "binary_sha256"] = "wrong"
    elif fault == "nonfinite": frame.loc[0, "I_p"] = float("inf")
    else: frame.loc[0, "task_id"] = "outside"
    write_table(collection, "indices.csv.gz", frame)
    manifest = json.loads((collection / "collection.json").read_text())
    manifest["files"]["analysis.sqlite"] = sha(collection / "analysis.sqlite")
    atomic_json(collection / "collection.json", manifest)
    with pytest.raises(ValueError): compare(prepared, collection, tmp_path / "bad")


def test_preparation_authentication_rejects_modified_output(tmp_path):
    prepared = tmp_path / "prepared"
    prepare([_bundle(tmp_path)], prepared)
    (prepared / "paths.csv").write_text('edited')
    with pytest.raises(ValueError, match="hash mismatch"):
        authenticate_preparation(prepared)


def test_alignment_preserves_an_explicit_integral_only_policy(tmp_path):
    prepared = tmp_path / "prepared"
    prepare([_bundle(tmp_path)], prepared)
    result = compare(prepared, _collection(tmp_path, prepared, complement=False), tmp_path / "comparison")
    assert result.endpoint_complement is False


def test_prepare_rejects_retired_review_layout(tmp_path):
    bundle = _bundle(tmp_path)
    saved = json.loads((bundle / "data/manifest.json").read_text())
    review = tmp_path / "review"
    review.mkdir()
    shutil.move(bundle / "data/study", review / "source")
    manifest = {key: value for key, value in saved.items() if key not in ("schema", "source_dataset", "files")}
    manifest["source"] = saved["source_dataset"]
    manifest["files"] = {key.removeprefix("study/"): value for key, value in saved["files"].items()}
    atomic_json(review / "source_manifest.json", manifest)
    with pytest.raises(FileNotFoundError):
        prepare([review], tmp_path / "prepared-review")
