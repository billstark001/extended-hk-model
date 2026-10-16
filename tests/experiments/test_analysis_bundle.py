"""CSV identity, incremental invalidation and genuine raw-free rendering boundaries."""

import json

import numpy as np
import pytest

from common.experiments.artifacts import ArtifactCache
from common.experiments.plans import freeze_plan
from common.io.records import read_record, write_record
from experiments.paper_theory.evidence.analyze import analyze
from experiments.paper_theory.evidence.bundle import Bundle, export_raw
from experiments.paper_theory.evidence.design import design
from tests.experiments.test_theory_named_tasks import synthetic_records
from theory.variation import components, with_loio


def test_variance_components_keep_negative_corrections_and_exact_identities():
    y = np.array([[[0.0], [2.0]], [[0.0], [2.0]], [[0.0], [2.0]], [[0.0], [2.0]]])
    p = np.ones((4, 1))
    result = components(y, p)
    assert result["process_variance"] == 2
    assert result["initial_variance_corrected"] == -1
    assert result["conditional_bias_squared_corrected"] == -1
    assert result["individual_mse"] == 1
    assert result["individual_mse"] == result["process_variance"] + result["conditional_bias_squared_corrected"]
    assert result["initial_variance_corrected"] == (
        result["prediction_initial_variance"]
        + result["residual_initial_variance_corrected"]
        + 2 * result["prediction_residual_covariance"]
    )
    rng = np.random.default_rng(31)
    y, p = rng.normal(size=(4, 4, 9)), rng.normal(size=(4, 9))
    baseline = with_loio(y, p, 0.2)
    changed = with_loio(-3 * y[::-1, ::-1] + 10, -3 * p[::-1] + 10, 0.2)
    for key, value in baseline.items():
        assert changed[key] == pytest.approx(9 * value, abs=1e-12)


def test_csv_export_roundtrips_all_native_arrays_and_authenticates_changes(tmp_path):
    plan = design("smoke_scaling")
    # Two numerical grids ensure native shape and dtype values are preserved.
    plan = {**plan, "tasks": plan["tasks"][:2]}
    raw = tmp_path / "raw_source"
    freeze_plan(raw / "plan.json", plan)
    synthetic_records(raw, plan)
    out = tmp_path / "local"
    export_raw(raw, out)
    bundle = Bundle(out)
    for task in plan["tasks"]:
        case = bundle.case(task)
        with np.load(raw / "measure" / "raw" / (task["id"] + ".npz"), allow_pickle=False) as original:
            for name in original.files:
                if name not in ("metadata", "diagnostics"):
                    np.testing.assert_array_equal(case[name], original[name])
    before = {k: v for k, v in bundle.manifest["files"].items()}
    export_raw(raw, out)
    after = Bundle(out)
    assert after.manifest["reused_records"] == 2
    assert after.manifest["files"] == before
    file = next(v["series"][0] for v in after.manifest["records"].values())
    (out / file).write_bytes(b"broken CSV")
    with pytest.raises(ValueError, match="CSV content changed"):
        Bundle(out).case(plan["tasks"][0])


def test_incremental_statistics_invalidate_only_changed_configuration_and_force_all(tmp_path):
    plan = design("smoke_population")
    raw = tmp_path / "source"
    freeze_plan(raw / "plan.json", plan)
    synthetic_records(raw, plan)
    out = tmp_path / "bundle"
    analyze(raw, out)
    baseline = json.loads((out / "analysis_manifest.json").read_text())
    count = baseline["cache"]["computed"]
    assert count > 1
    analyze(raw, out)
    second = json.loads((out / "analysis_manifest.json").read_text())
    assert second["cache"] == {"computed": 0, "reused": count, "forced": False}
    # One scientific record changes; unrelated configurations and pair CSVs survive.
    task = plan["tasks"][0]
    path = raw / "micro" / "raw" / (task["id"] + ".msgpack.lz4")
    record = read_record(path)
    record["result"]["frames"][1]["exact_node_energy"] *= 0.99
    write_record(path, task, record["result"])
    analyze(raw, out)
    changed = json.loads((out / "analysis_manifest.json").read_text())
    assert 0 < changed["cache"]["computed"] < count
    assert changed["cache"]["reused"] > 0
    hashes = changed["tables"]
    # Explicit full recomputation also works without the raw volume.
    raw.rename(tmp_path / "disconnected")
    analyze(out, force=True)
    forced = json.loads((out / "analysis_manifest.json").read_text())
    assert forced["cache"]["forced"] and forced["cache"]["computed"] == count
    assert forced["cache"]["reused"] == 0
    assert forced["tables"] == hashes


def test_record_metadata_changes_invalidate_statistics_without_csv_changes(tmp_path):
    plan = design("smoke_scaling")
    plan = {**plan, "tasks": plan["tasks"][:2]}
    raw, out = tmp_path / "raw", tmp_path / "local"
    freeze_plan(raw / "plan.json", plan)
    synthetic_records(raw, plan)
    assert analyze(raw, out)["early_stopped_measure"] == 0
    manifest = json.loads((out / "data_manifest.json").read_text())
    manifest["records"][plan["tasks"][0]["id"]]["diagnostics"]["converged"] = True
    (out / "data_manifest.json").write_text(json.dumps(manifest))
    assert analyze(out)["early_stopped_measure"] == 1
    updated = json.loads((out / "analysis_manifest.json").read_text())
    assert updated["cache"]["computed"] == 1
    assert json.loads((out / "data_manifest.json").read_text())["files"] == manifest["files"]


def test_artifact_cache_never_commits_failed_or_corrupt_computations(tmp_path):
    cache = ArtifactCache(tmp_path / "cache")
    calls = []

    def producer(path):
        calls.append(1)
        (path / "values.csv").write_text("value\n2\n")

    path = cache.materialize("one", {"source": "a"}, producer)
    assert cache.materialize("one", {"source": "a"}, producer) == path
    assert len(calls) == 1
    (path / "values.csv").write_text("changed")
    cache.materialize("one", {"source": "a"}, producer)
    assert len(calls) == 2

    def failed(path):
        (path / "partial.csv").write_text("unfinished")
        raise ValueError("interrupted")

    with pytest.raises(ValueError):
        cache.materialize("two", {"source": "a"}, failed)
    assert not list((tmp_path / "cache" / "two").rglob("artifact.json"))
