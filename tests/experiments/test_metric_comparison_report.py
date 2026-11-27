"""Scientific reports preserve paired states, pathway prefixes and metric meanings."""

import json
import subprocess

import pandas as pd
import pytest

from common.io.json import atomic_json, read_result, save_result, sha
from experiments.paper.metric_comparison.core import PATH_KEYS, prepare
from experiments.paper.metric_comparison.report import (
    STATE_METRICS,
    checkpoint_table,
    effect_summaries,
    endpoint_features,
    endpoint_table,
    paired_table,
    pathway_table,
    peak_summaries,
    report,
)


@pytest.fixture(autouse=True)
def forbid_simulator(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("scientific reporting must not launch a process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def _tables():
    points, paths = [], []
    def add(task, stage, arm, states, *, influence=.01, rewiring=.1, consensus=False, iw=.3, profile="staged"):
        keys = dict(profile=profile, task_id=task, stage=stage, arm=arm)
        for step, p, e in states:
            # High Ih is intentionally retained in consensus: it measures homophily.
            ih = 1.0 if consensus and step == states[-1][0] else .5 + e
            points.append({**keys, "step": step, "relative_step": step-states[0][0],
                           "separation": p, "excess_concordance": e, "edge_concordance": .5+e,
                           "pair_concordance": .5, "I_p": p, "I_h": ih, "I_s": .4})
        paths.append({**keys, "start_step": states[0][0], "end_step": states[-1][0],
                      "I_w": iw, "terminal_json": json.dumps({"status": "absorbed", "k_major": 1 if consensus else 2}),
                      "influence": influence, "rewiring": rewiring, "repost": .1,
                      "sensitivity": "core", "epsilon": .45, "history": 0})
    for task, middle, final, alpha, q in (
        ("a", (.5, .05), (.8, .2), .01, .1),
        ("b", (.3, .15), (.7, .2), .1, .01),
        ("c", (.5, .05), (.6, .2), .05, .05),
    ):
        baseline = [(0, .1, 0), (16, *middle), (32, *final)]
        add(task, "reference", "reference", baseline, influence=alpha, rewiring=q)
        add(task, "early", "hold", baseline[:2], influence=alpha, rewiring=q)
        add(task, "early", "switch", [(0, .1, 0), (16, middle[0]+.1, middle[1])], influence=alpha, rewiring=q)
        add(task, "middle", "hold", baseline[1:], influence=alpha, rewiring=q)
        changed = 0 if task == "a" else final[0]+.02
        add(task, "middle", "switch", [(16, *middle), (32, changed, 0 if task == "a" else final[1])],
            influence=alpha, rewiring=q, consensus=task=="a")
    for arm, final in (("hold", .5), ("fixed", .4), ("state", .6)):
        add("d", "full", arm, [(0, .1, 0), (16, .3, .1), (32, final, .2)], profile="staged_timing")
    return pd.DataFrame(points), pd.DataFrame(paths)


def test_existing_first_passage_uses_common_prefix_not_intervention_window():
    points, paths = _tables()
    endpoints = endpoint_table(points, paths)
    result = pathway_table(points, endpoints)
    a = result[(result.task_id == "a") & (result.stage == "middle") & (result.arm == "switch")].iloc[0]
    assert a.pathway == "separation_only"  # P crossed at the shared checkpoint; switch prevents E crossing.
    assert a.separation_step == 16
    assert a.prefix_observations == 1
    assert result[(result.task_id == "a") & (result.stage == "reference")].iloc[0].pathway == "PbS"
    assert result[(result.task_id == "b") & (result.stage == "reference")].iloc[0].pathway == "SbP"


def test_middle_average_is_explicitly_decomposed_into_consensus_gains_and_remaining():
    points, paths = _tables()
    endpoints = endpoint_table(points, paths)
    pairs = paired_table(endpoints)
    effects = effect_summaries(pairs)
    selected = effects[effects.metric == "separation"].set_index("scope")
    assert selected.loc["early_q_gt_alpha", "n"] == 1
    assert selected.loc["middle_P_ge_0.4_E_lt_0.1", "n"] == 2
    assert selected.loc["middle_P_ge_0.4_E_lt_0.1", "mean"] == pytest.approx(-.39)
    assert selected.loc["middle_consensus_gains", "sum_delta"] == pytest.approx(-.8)
    assert selected.loc["middle_remaining", "sum_delta"] == pytest.approx(.02)
    # The result must preserve high homophily after consensus instead of forcing metric agreement.
    state = endpoints[(endpoints.task_id == "a") & (endpoints.stage == "middle") & (endpoints.arm == "switch")].iloc[0]
    assert state.end_separation == 0 and state.end_I_h == 1


def test_pairing_rejects_different_absolute_endpoints():
    points, paths = _tables()
    endpoints = endpoint_table(points, paths)
    endpoints.loc[(endpoints.task_id == "a") & (endpoints.stage == "middle") & (endpoints.arm == "switch"), "end_step"] = 48
    with pytest.raises(ValueError, match="end_step"):
        paired_table(endpoints)


def test_pathway_rejects_missing_reference_prefix_and_16_step_gaps():
    points, paths = _tables()
    endpoints = endpoint_table(points, paths)
    without = points[~((points.task_id == "a") & (points.stage == "reference"))]
    with pytest.raises(ValueError, match="reference prefix"):
        pathway_table(without, endpoints[~((endpoints.task_id == "a") & (endpoints.stage == "reference"))])
    incomplete = points[~((points.task_id == "a") & (points.stage == "reference") & (points.step == 16))]
    with pytest.raises(ValueError, match="16-step grid"):
        pathway_table(incomplete, endpoints)


def test_partial_metrics_use_the_identical_available_pairs_for_every_metric():
    points, paths = _tables()
    points.loc[(points.task_id == "a") & (points.stage == "middle") & (points.arm == "switch"), "I_p"] = float("nan")
    pairs = paired_table(endpoint_table(points, paths))
    effects = effect_summaries(pairs)
    rows = effects[effects.scope == "middle_P_ge_0.4_E_lt_0.1"]
    assert set(rows.n) == {1}
    assert set(rows.pending_pairs) == {1}
    assert set(rows.metric) == set(STATE_METRICS)


def _evidence(tmp_path, *, status="complete"):
    points, paths = _tables()
    prep, compared = tmp_path / "prepared", tmp_path / "compared"
    prep.mkdir()
    compared.mkdir()
    points.to_csv(prep / "points.csv.gz", index=False)
    paths.to_csv(prep / "paths.csv", index=False)
    atomic_json(prep / "preparation.json", {
        "schema": 1, "points": len(points), "paths": len(paths),
        "files": {p.name: sha(p) for p in prep.iterdir()},
    })
    points.to_csv(compared / "points_aligned.csv.gz", index=False)
    paths.to_csv(compared / "paths_aligned.csv.gz", index=False)
    atomic_json(compared / "comparison.json", {
        "schema": 1, "status": status, "preparation_sha256": sha(prep / "preparation.json"),
        "files": {p.name: sha(p) for p in compared.iterdir()},
    })
    return prep, compared


def test_report_authenticates_and_renders_scientific_artifacts(tmp_path, monkeypatch):
    # Artifact layout is independent of the authenticated peak-counter tests below.
    def features(preparation, endpoints):
        result = endpoints[[*PATH_KEYS, "end_step"]].copy()
        result["opinion_peaks"] = 2
        result["saved_count_matches"] = None
        return result
    monkeypatch.setattr("experiments.paper.metric_comparison.report.endpoint_features", features)
    prep, compared = _evidence(tmp_path)
    output = tmp_path / "report"
    result = report(prep, compared, output)
    assert result["status"] == "complete"
    assert (output / "report.md").is_file()
    assert "middle_consensus_gains" in (output / "report.md").read_text(encoding="utf-8")
    assert (output / "checkpoint_states.csv").is_file()
    assert (output / "kde_confidence_crosstab.csv").is_file()
    assert (output / "kde_stratified_states.csv").is_file()
    for stem in ("matched_states", "checkpoint_states", "pathway_metrics", "intervention_effects"):
        for suffix in ("png", "pdf", "svg"):
            assert (output / f"{stem}.{suffix}").stat().st_size > 100


def test_pending_comparison_requires_explicit_interim_report(tmp_path):
    prep, compared = _evidence(tmp_path, status="pending_legacy_metrics")
    with pytest.raises(ValueError, match="pending"):
        report(prep, compared, tmp_path / "report", make_figures=False)
    assert not (tmp_path / "report").exists()


def test_start_roundoff_validation_retains_saved_values_and_rejects_real_differences():
    points, paths = _tables()
    mask = (points.task_id == "a") & (points.stage == "middle") & (points.arm == "switch") & (points.step == 16)
    points.loc[mask, "I_s"] += 2e-16
    endpoints = endpoint_table(points, paths)
    paired = paired_table(endpoints)
    row = paired[(paired.task_id == "a") & (paired.stage == "middle")].iloc[0]
    assert 0 < row.start_difference_I_s < 1e-15
    assert endpoints[(endpoints.task_id == "a") & (endpoints.stage == "middle") & (endpoints.arm == "switch")].iloc[0].start_I_s > .4
    points.loc[mask, "I_s"] += 1e-5
    with pytest.raises(ValueError, match="I_s"):
        paired_table(endpoint_table(points, paths))


def test_checkpoint_table_keeps_one_hold_start_per_task_and_stage():
    points, paths = _tables()
    checkpoints = checkpoint_table(endpoint_table(points, paths))
    assert not checkpoints.duplicated(["profile", "task_id", "stage"]).any()
    assert len(checkpoints) == 7
    a = checkpoints[(checkpoints.task_id == "a") & (checkpoints.stage == "middle")].iloc[0]
    assert a.step == 16 and a.separation == .5 and a.excess_concordance == .05
    assert a.arm == "hold"


def test_consensus_need_not_have_a_low_legacy_objective_index():
    points, paths = _tables()
    selected = (points.task_id == "a") & (points.stage == "middle") & (points.arm == "switch") & (points.step == 32)
    points.loc[selected, "I_p"] = 1.4
    pairs = paired_table(endpoint_table(points, paths))
    row = pairs[(pairs.task_id == "a") & (pairs.stage == "middle")].iloc[0]
    assert row.consensus_gain
    assert row.treatment_I_p == 1.4
    assert row.delta_separation < 0 and row.delta_I_p > 0


def _peak_source(tmp_path, *, archived_count=2):
    from tests.experiments.test_metric_comparison import _bundle
    bundle = _bundle(tmp_path)
    data = bundle / "data"
    record_path = next((data / "study/records").glob("*.json.gz"))
    record = read_result(record_path)
    record["result"]["reference"]["final_opinions"] = [-.1] * 500
    record["result"]["stages"]["middle"]["arms"]["switch"]["final_opinions"] = [-.8] * 400 + [.8] * 100
    save_result(record_path, record)
    task_id = record_path.name.removesuffix(".json.gz")
    (data / "tables").mkdir()
    pd.DataFrame([{"task_id": task_id, "stage": "middle", "arm": "switch", "end_step": 32,
                   "opinion_peaks": archived_count}]).to_csv(data / "tables/runs.csv.gz", index=False)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = {p.relative_to(data).as_posix(): sha(p) for p in data.rglob("*")
                         if p.is_file() and p.name != "manifest.json"}
    atomic_json(data / "manifest.json", manifest)
    prepared = tmp_path / "prepared-peaks"
    prepare([bundle], prepared)
    paths = pd.read_csv(prepared / "paths.csv", float_precision="round_trip")
    return prepared, paths, record_path


def test_existing_kde_counts_include_reference_paths_and_recheck_saved_counts(tmp_path):
    prepared, paths, _ = _peak_source(tmp_path)
    features = endpoint_features(prepared, paths).set_index("stage")
    assert features.loc["reference", "opinion_peaks"] == 1
    assert pd.isna(features.loc["reference", "saved_count_matches"])
    assert features.loc["middle", "opinion_peaks"] == 2
    assert features.loc["middle", "saved_count_matches"]
    assert features["summary_sha256"].str.len().eq(64).all()


def test_peak_counts_keep_archived_disagreement_without_a_truth_label(tmp_path):
    prepared, paths, _ = _peak_source(tmp_path, archived_count=1)
    features = endpoint_features(prepared, paths)
    row = features[features.stage == "middle"].iloc[0]
    assert row.opinion_peaks == 2 and row.saved_opinion_peaks == 1
    assert not row.saved_count_matches
    points, endpoints = _tables()
    endpoints = endpoint_table(points, endpoints)
    endpoints["opinion_peaks"] = 2
    cross, summary = peak_summaries(endpoints)
    assert "consensus" in set(cross.terminal_outcome)
    assert set(cross.opinion_peaks) == {2}
    assert not any("accuracy" in name or "truth" in name for name in cross)
    assert set(summary.metric) == {"I_p", "I_h", "separation", "excess_concordance"}


def test_endpoint_feature_join_rejects_horizon_mismatch_and_tampered_source(tmp_path):
    prepared, paths, record_path = _peak_source(tmp_path)
    wrong = paths.copy()
    wrong.loc[wrong.stage == "middle", "end_step"] += 16
    with pytest.raises(ValueError, match="horizon"):
        endpoint_features(prepared, wrong)
    record_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        endpoint_features(prepared, paths)
