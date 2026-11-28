"""Frozen-identity and completeness boundaries for offline comparison reports."""

import csv
import itertools
import json
from pathlib import Path

import pytest

from common.io.json import fingerprint, sha
from common.storage.documents import DocumentStore
from ehk.metrics import pathway
from experiments.paper.pathway_metrics import report


def freeze_study(tmp_path, *, completed=8):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    source = tmp_path / "comparisons.sqlite"
    core = Path(pathway.__file__).resolve()
    protocol = dict(schema=1, observation_interval=16, include_terminal=True, separation_threshold=.4,
                    excess_threshold=.1, interpretation="Fixture diagnostic", probe_binary="not-executed",
                    probe_sha256="fixture", source_sha256={str(core): sha(core)},
                    input_reader="fixture saved observations", inline_batch_size=16)
    protocol_id = fingerprint(protocol)
    design = dict(analysis_id="adopted-primary", alpha_axis=[.01, .1], q_axis=[.01, .1],
                  p_axis=[0.], algorithms=["StructureM9", "OpinionM9"],
                  post_retain_count=0, pe_observation_interval=16,
                  reference_pe_thresholds={"P": .4, "E": .1})
    identities, tasks, results = [], [], []
    # All physically available order categories, including trajectories that
    # have reached neither or only one threshold; unavailable stays explicit 0.
    histories = [((0., 0.), (.5, 0.), (.5, .2)),
                 ((0., 0.), (0., .2), (.5, .2)),
                 ((0., 0.), (.5, .2), (.5, .2)),
                 ((0., 0.), (.5, 0.), (.5, 0.)),
                 ((0., 0.), (0., .2), (0., .2)),
                 ((0., 0.), (0., 0.), (0., 0.))]
    cells = itertools.product(design["algorithms"], design["alpha_axis"], design["q_axis"], design["p_axis"])
    for i, (algorithm, alpha, q, p) in enumerate(cells):
        identity = dict(origin="data2" if i % 2 == 0 else "data3", name=f"run-{i}")
        primary = dict(identity=identity, analysis_id="adopted-primary", metric_revision=6,
                       endpoint_complement=True, source_fingerprint=f"raw-{i}",
                       metadata=dict(HKParams=dict(Influence=alpha, RewiringRate=q, RepostRate=p),
                                     RecsysFactoryType=algorithm, PostRetainCount=0, MaxSimulationStep=32),
                       terminal_step=32, I_w=.8 if i % 2 else .2,
                       terminal_I_p=.25, terminal_I_h=.8, opinion_peaks=1 if i % 2 else 2)
        key = fingerprint((protocol_id, primary))
        observations = [dict(step=t, P=pv, E=ev, C_edge=.8, C_pair=.4)
                        for t, (pv, ev) in zip((0, 16, 32), histories[i % 6])]
        passage = pathway.first_passage((o["step"], o["P"], o["E"]) for o in observations)
        value = dict(schema=1, state="complete", protocol_id=protocol_id, primary=primary,
                     observations=observations, observed_order=passage.label,
                     separation_step=passage.separation_step, segregation_step=passage.segregation_step,
                     error=None)
        identities.append(identity)
        tasks.append(dict(identity=identity, result_key=key, primary_sha256=fingerprint(primary),
                          source_fingerprint=primary["source_fingerprint"], raw_directory="not-read",
                          terminal_step=32, expected_observations=3, saved_graph_steps=[0, 32]))
        results.append((key, value))
    selection = dict(schema=1, design=design, tasks=identities)
    selection_path = study_dir / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    study = dict(schema=1, study_id="fixture-study", primary_database="not-read", comparison_database=str(source), analysis_id="adopted-primary",
                 selection_file="selection.json", selection_sha256=sha(selection_path),
                 selection_source=str(selection_path), selection_source_sha256=sha(selection_path),
                 protocol_id=protocol_id, protocol=protocol, plan_id="fixture-plan", tasks=tasks,
                 adopted_complete_keys=[])
    study["manifest_sha256"] = fingerprint(study)
    (study_dir / "study.json").write_text(json.dumps(study), encoding="utf-8")
    with DocumentStore(source) as store:
        for key, result in results[:completed]:
            store.put(report.NAMESPACE, key, result)
        # A stale result in another protocol must not enter any denominator.
        foreign = dict(results[0][1], protocol_id="foreign")
        store.put(report.NAMESPACE, "not-selected", foreign)
    return study_dir, source, results


def test_preliminary_report_keeps_categories_and_commits_sqlite_before_exports(tmp_path):
    study, source, _ = freeze_study(tmp_path, completed=6)
    before = sha(source)
    output = tmp_path / "preliminary"
    result = report.analyze(study, output, preliminary=True, make_figure=False)
    assert (result["complete"], result["planned"], result["missing"]) == (6, 8, 2)
    assert sha(source) == before
    with DocumentStore(output / "analysis.sqlite", readonly=True) as store:
        receipt = store.get(report.REPORTS, result["report_id"])
        table = store.get(report.TABLES, receipt["tables"]["counts"])["rows"]
        overall = [r for r in table if r["dimension"] == "overall"]
        assert {r["PE_label"] for r in overall} == set(report.LABELS)
        assert sum(r["n"] for r in overall) == 6
        assert sum(r["n"] for r in overall if r["PE_label"] == "unavailable") == 0
        thresholds = store.get(report.TABLES, receipt["tables"]["thresholds"])["rows"]
        split = next(r for r in thresholds if r["dimension"] == "overall" and r["cutoff"] == .6)
        assert split["ordered_both_thresholds_n"] == 2
        assert split["both_thresholds_observed_n"] == 3
        assert split["simultaneous_n"] == 1
    with (output / "counts.csv").open(encoding="utf-8", newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == len(table)
    # SQLite export must not reopen source observations, design inputs or CSV.
    (output / "counts.csv").unlink()
    source.unlink()
    (study / "selection.json").unlink()
    (study / "study.json").unlink()
    report.export_report(output / "analysis.sqlite", result["report_id"], output, make_figure=True)
    assert (output / "counts.csv").is_file()
    assert (output / "comparison.png").is_file()
    assert (output / "comparison.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "comparison_caption.txt").is_file()


def test_incomplete_default_refuses_before_creating_report(tmp_path):
    study, _, _ = freeze_study(tmp_path, completed=6)
    output = tmp_path / "complete"
    with pytest.raises(ValueError, match="Incomplete selected sample"):
        report.analyze(study, output, make_figure=False)
    assert not output.exists()


def test_frozen_primary_reference_change_is_rejected(tmp_path):
    study, source, results = freeze_study(tmp_path)
    key, changed = results[0]
    changed["primary"]["I_w"] = .9
    with DocumentStore(source) as store:
        store.put(report.NAMESPACE, key, changed)
    with pytest.raises(ValueError, match="primary reference differs"):
        report.analyze(study, tmp_path / "report", make_figure=False)


def test_manifest_content_change_without_reauthentication_is_rejected(tmp_path):
    study, _, _ = freeze_study(tmp_path)
    path = study / "study.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["comparison_database"] = str(tmp_path / "another.sqlite")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest content/hash mismatch"):
        report.analyze(study, tmp_path / "report", make_figure=False)


def test_task_pinned_protocol_retains_authenticated_existing_results(tmp_path):
    study, source, results = freeze_study(tmp_path)
    value = results[0][1]
    value["protocol_id"] = "retained-task-protocol"
    key = fingerprint((value["protocol_id"], value["primary"]))
    with DocumentStore(source) as store:
        store.put(report.NAMESPACE, key, value)
    path = study / "study.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["tasks"][0].update(result_key=key, result_protocol_id=value["protocol_id"])
    manifest["manifest_sha256"] = fingerprint({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = report.analyze(study, tmp_path / "retained", make_figure=False)
    assert result["complete"] == 8
    with DocumentStore(Path(result["analysis_database"]), readonly=True) as store:
        receipt = store.get(report.REPORTS, result["report_id"])
    assert receipt["result_protocol_counts"]["retained-task-protocol"] == 1


def test_complete_report_preserves_endpoint_algebra_and_cap_counts(tmp_path):
    study, _, _ = freeze_study(tmp_path)
    output = tmp_path / "complete"
    result = report.analyze(study, output, make_figure=False)
    assert result["complete"] == result["planned"] == 8
    assert result["preliminary"] is False
    with DocumentStore(output / "analysis.sqlite", readonly=True) as store:
        receipt = store.get(report.REPORTS, result["report_id"])
        endpoints = store.get(report.TABLES, receipt["tables"]["endpoints"])["rows"]
        assert all(r["median_endpoint_completion"] == pytest.approx(.6) for r in endpoints if r["n"])
        horizons = store.get(report.TABLES, receipt["tables"]["horizons"])["rows"]
        overall = next(r for r in horizons if r["dimension"] == "overall" and r["PE_label"] == "all")
        assert overall["at_simulation_cap_n"] == 8


def test_reexport_records_renderer_separately_without_changing_analysis(tmp_path):
    study, _, _ = freeze_study(tmp_path)
    output = tmp_path / "analysis"
    result = report.analyze(study, output, make_figure=False)
    database = output / "analysis.sqlite"
    before = sha(database)
    with DocumentStore(database, readonly=True) as store:
        receipt = store.get(report.REPORTS, result["report_id"])
    exported = tmp_path / "restyled"
    report.export_report(database, result["report_id"], exported, make_figure=False)
    with (exported / "provenance.csv").open(encoding="utf-8", newline="") as stream:
        provenance = {r["property"]: r["value"] for r in csv.DictReader(stream)}
    assert sha(database) == before
    assert provenance["analysis_receipt_sha256"] == fingerprint(receipt)
    assert provenance["report_script_sha256"] == receipt["report_script_sha256"]
    assert provenance["renderer_source_sha256"] == sha(report.__file__)
    assert len(provenance["plotting_core_source_sha256"]) == 64
    assert json.loads(provenance["render_parameters"])["width_inches"] == 8
    assert json.loads(provenance["render_parameters"])["dpi"] == 300
