"""Authenticated alignment preserves unavailable metrics and exact paired identities."""

import json
import shutil
import subprocess

import pandas as pd
import pytest

from common.experiments.plans import freeze_plan
from common.io.json import atomic_json, fingerprint, save_result, sha
from experiments.paper.metric_comparison.core import (
    IW_SAMPLING_METHOD,
    LEGACY_PATHS,
    LEGACY_POINTS,
    PATH_KEYS,
    POINT_KEYS,
    PROVENANCE,
    authenticate_preparation,
    collect,
    compare,
    prepare,
)
from experiments.paper.recommendation.design import generate_tasks


@pytest.fixture(autouse=True)
def forbid_native(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("metric preparation must not start native processes")
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def _bundle(tmp_path):
    root = tmp_path / "bundle"
    data = root / "data"
    task = generate_tasks("staged")[0]
    plan = freeze_plan(data / "study/plan.json", {
        "schema": 3, "study": "paper_recommendation", "profile": "staged", "protocol": {}, "tasks": [task],
    })
    def point(step):
        return {"step": step, "relative_step": step,
                "pathway": {"separation": .2, "excess_concordance": .1,
                            "edge_concordance": .6, "pair_concordance": .5}}
    def arm(stage, name, steps):
        return {"raw_record": f"raw/{task['id']}/{stage}/{name}", "endpoints": [point(s) for s in steps],
                "terminal": {"status": "nonterminal", "k_major": 1}}
    record = {"reference": arm("reference", "reference", [0, 16]),
              "stages": {"middle": {"arms": {"switch": arm("middle", "switch", [16, 32])}}}}
    save_result(data / f"study/records/{task['id']}.json.gz", {
        "task_fingerprint": fingerprint(task), "result": record,
    })
    atomic_json(data / "study/runtime/binding.json", {"binary_sha256": "original-binary", "stage_evidence": None})
    manifest = {"schema": 1, "planned": 1, "completed": 1, "plan_fingerprint": plan["plan_fingerprint"],
                "source_dataset": str(tmp_path / "unmounted"),
                "files": {p.relative_to(data).as_posix(): sha(p) for p in data.rglob("*") if p.is_file()}}
    atomic_json(data / "manifest.json", manifest)
    return root


def test_prepare_exports_reference_and_branch_with_pending_raw(tmp_path):
    bundle = _bundle(tmp_path)
    output = tmp_path / "prepared"
    before = {p: sha(p) for p in bundle.rglob("*") if p.is_file()}
    report = prepare([bundle], output)
    assert (report["points"], report["paths"], report["pending_raw_paths"]) == (4, 2, 2)
    assert report["missing_metrics"] == ["I_p", "I_s", "I_h", "I_w"]
    points = pd.read_csv(output / "points.csv.gz")
    paths = pd.read_csv(output / "paths.csv")
    assert not points.duplicated(POINT_KEYS).any()
    assert set(points["stage"]) == {"reference", "middle"}
    assert points["excess_concordance"].tolist() == [.1] * 4
    assert paths["full_prefix_required"].all()
    assert not paths["available"].any()
    assert paths.loc[paths["stage"] == "middle", "start_step"].item() == 16
    assert json.loads(paths.iloc[0]["metadata_json"])["RNG"] == json.loads(paths.iloc[0]["seed_json"])
    assert authenticate_preparation(output)["points"] == 4
    assert all(sha(p) == digest for p, digest in before.items())


def test_prepare_rejects_tampered_manifest_and_duplicate_generations(tmp_path):
    bundle = _bundle(tmp_path)
    with pytest.raises(ValueError, match="duplicate task"):
        prepare([bundle, bundle], tmp_path / "duplicates")
    binding = bundle / "data/study/runtime/binding.json"
    binding.write_text('{}')
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare([bundle], tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def _legacy(tmp_path, preparation):
    points = pd.read_csv(preparation / "points.csv.gz", keep_default_na=False).iloc[:1][[*POINT_KEYS, *PROVENANCE]].copy()
    for name, value in zip(LEGACY_POINTS, [.3, .4, .5]):
        points[name] = value
    paths = pd.read_csv(preparation / "paths.csv", keep_default_na=False).iloc[:1].copy()
    paths["I_w"], paths["sampling_method"], paths["prefix_start"] = .7, IW_SAMPLING_METHOD, 0
    paths["trajectory_end"] = paths["end_step"]
    paths["endpoint_complement"] = True
    paths = paths[[*PATH_KEYS, *PROVENANCE, *LEGACY_PATHS]]
    points.to_csv(tmp_path / "old-points.csv", index=False)
    paths.to_csv(tmp_path / "old-paths.csv", index=False)
    return points, paths


def test_exact_import_and_join_leave_missing_values_pending(tmp_path):
    prepared = tmp_path / "prepared"
    prepare([_bundle(tmp_path)], prepared)
    _legacy(tmp_path, prepared)
    imported = tmp_path / "imported"
    report = collect(prepared, imported, points=tmp_path / "old-points.csv", paths=tmp_path / "old-paths.csv")
    assert report["status"] == "pending_legacy_metrics"
    report = compare(prepared, imported, tmp_path / "comparison")
    assert report["coverage"] == {"points": {"requested": 4, "matched": 1, "pending": 3},
                                   "paths": {"requested": 2, "matched": 1, "pending": 1}}
    aligned = pd.read_csv(tmp_path / "comparison/points_aligned.csv.gz")
    assert aligned["I_p"].isna().sum() == 3
    assert aligned.loc[aligned["legacy_coverage"] == "both", "I_h"].item() == .5
    assert aligned["separation"].tolist() == [.2] * 4


@pytest.mark.parametrize("problem", ["duplicate", "off_grid", "provenance", "prefix", "nonfinite", "16_step_integral"])
def test_import_rejects_inexact_or_unusable_legacy_data(tmp_path, problem):
    prepared = tmp_path / "prepared"
    prepare([_bundle(tmp_path)], prepared)
    points, paths = _legacy(tmp_path, prepared)
    if problem == "duplicate":
        points = pd.concat([points, points])
    elif problem == "off_grid":
        points.loc[points.index[0], "step"] = 8
    elif problem == "provenance":
        points.loc[points.index[0], "binary_sha256"] = "other"
    elif problem == "prefix":
        paths.loc[paths.index[0], "prefix_start"] = 16
    elif problem == "16_step_integral":
        paths.loc[paths.index[0], "sampling_method"] = "16_step_trapezoid"
    else:
        points.loc[points.index[0], "I_p"] = float('nan')
    points.to_csv(tmp_path / "old-points.csv", index=False)
    paths.to_csv(tmp_path / "old-paths.csv", index=False)
    with pytest.raises(ValueError):
        collect(prepared, tmp_path / "bad-import", points=tmp_path / "old-points.csv", paths=tmp_path / "old-paths.csv")


def test_preparation_authentication_rejects_modified_output(tmp_path):
    prepared = tmp_path / "prepared"
    prepare([_bundle(tmp_path)], prepared)
    (prepared / "paths.csv").write_text('edited')
    with pytest.raises(ValueError, match="hash mismatch"):
        authenticate_preparation(prepared)


def test_historical_import_requires_explicit_integral_only_policy(tmp_path):
    prepared = tmp_path / "prepared"
    prepare([_bundle(tmp_path)], prepared)
    _, paths = _legacy(tmp_path, prepared)
    historical = tmp_path / "old-paths.csv"
    paths.drop(columns="endpoint_complement").to_csv(historical, index=False)
    with pytest.raises(ValueError, match="no endpoint_complement policy"):
        collect(prepared, tmp_path / "ambiguous", paths=historical)
    imported = tmp_path / "historical"
    result = collect(prepared, imported, paths=historical, endpoint_complement=False)
    assert result["endpoint_complement"] is False
    saved = pd.read_csv(imported / "legacy_paths.csv")
    assert saved["I_w"].item() == .7 and not saved["endpoint_complement"].item()
    comparison = compare(prepared, imported, tmp_path / "historical-comparison")
    assert comparison["endpoint_complement"] is False
    with pytest.raises(ValueError, match="differs from the selected policy"):
        collect(prepared, tmp_path / "relabelled", collection=imported)


def test_prepare_authenticates_legacy_review_layout(tmp_path):
    bundle = _bundle(tmp_path)
    saved = json.loads((bundle / "data/manifest.json").read_text(encoding="utf-8"))
    review = tmp_path / "review"
    review.mkdir()
    shutil.move(bundle / "data/study", review / "source")
    manifest = {key: value for key, value in saved.items() if key not in ("schema", "source_dataset", "files")}
    manifest["source"] = saved["source_dataset"]
    manifest["files"] = {key.removeprefix("study/"): value for key, value in saved["files"].items()}
    atomic_json(review / "source_manifest.json", manifest)
    report = prepare([review], tmp_path / "prepared-review")
    assert (report["points"], report["paths"]) == (4, 2)
