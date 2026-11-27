"""Portable figure stages preserve paired units and require no raw-data paths."""

import json
import shutil
from dataclasses import replace

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ehk.domain import Binding, FullResult, freeze_plan, to_wire
from tests.experiments.recommendation_fixtures import (
    arm as fixture_arm, point as fixture_point, exposure, plan as fixture_plan, stage, staged,
    bundle_manifest, report as fixture_report)
from experiments.paper.recommendation.figure_models import (
    BridgePanel, CaseCurve, CasePanel, IllustrativeCase, PairedEffectRow,
    PanelSpecifications, PlotLayout, PlotStyle, Thresholds)
from experiments.paper.recommendation.figure_models import parse_panel
from common.io.json import atomic_json, sha
from common.storage.documents import DocumentStore
from experiments.paper.recommendation import manuscript
from experiments.paper.recommendation.design import DESIGN_VERSION, generate_tasks
from experiments.paper.recommendation_shared import bundle as shared_bundle
from experiments.paper.recommendation_shared.analysis import (
    EXPOSURE_COUNTS,
    SLOT_MEASURES,
    SLOT_SOURCES,
    paired_effects,
)
from experiments.paper.recommender_variants import design as variants_design
from experiments.paper.recommender_variants.design import variant_arms
from experiments.paper.recommendation_shared.storage import DATABASE, StudyStore


def _point(step, start):
    return fixture_point(step, start, counters=replace(exposure(), updates=0 if step == start else 100))


def _arm(start):
    return fixture_arm(endpoints=(_point(start, start), _point(start+512, start)), raw_record="/unmounted/test/raw")


def _binding(source):
    atomic_json(source / "runtime/binding.json", Binding(binary_sha256="a" * 64))


def _specs(panels):
    return PanelSpecifications(schema=1, data_manifest_sha256="fixture", completed=1, planned=1,
        core_completed=1, core_planned=1, source_dataset="fixture", plan_fingerprint="fixture",
        coverage=(), panels=panels, thresholds=Thresholds(separation=.4, excess_concordance=.1),
        comparison_unit="paired initial state")


def _layout(*, marker_size=3.3, colors=None):
    return PlotLayout(schema=1, style=PlotStyle(marker_size=marker_size,
                      colors=manuscript.COLORS if colors is None else colors), figures={})


def _effect(*, delta_separation=0., delta_excess=0., replicate=0, repost=0., bridge_shown=0, updates=1000):
    return PairedEffectRow(task_id="fixture", stage="late", setting="fixture", sensitivity="core",
        influence=.1, rewiring=.01, repost=repost, replicate=replicate, epsilon=.45, history=0,
        bridge_shown=bridge_shown, cross_confident=0, updates=updates,
        delta_separation=delta_separation, delta_excess=delta_excess)


def test_export_then_compose_without_source_bundle_or_panel_images(tmp_path, monkeypatch):
    source, bundle, output = (tmp_path / name for name in ("source", "bundle", "paper"))
    source.mkdir()
    _binding(source)
    task = generate_tasks("staged")[0]
    plan = freeze_plan(source / "plan.json", fixture_plan(tasks=(task,)))
    record = staged()
    for phase, start in (("early", 0), ("middle", 16), ("late", 128)):
        record.stages[phase] = stage(start, {name: _arm(start) for name in ("hold", "switch", "bridge")})
    with StudyStore(source) as store:
        store.put_task(task, record, plan=plan)
    manifest = shared_bundle.export_bundle(source, bundle)
    assert manifest.completed == manifest.planned == 1
    shutil.rmtree(source)
    monkeypatch.setattr(manuscript, "CASES", (IllustrativeCase("fixture", task.id, "late", "fixture"),))
    manuscript.prepare(bundle, output)
    assert not (output / "panels").exists()
    assert not (output / "figures").exists()
    layout = json.loads((output / "layout.json").read_text())
    layout["figures"]["staged_intervention_cases"]["width"] = 6.8
    atomic_json(output / "layout.json", layout)
    manuscript.prepare(bundle, output)
    assert json.loads((output / "layout.json").read_text()) == layout
    shutil.rmtree(bundle)
    data_path = output / "plot_data.json"
    manuscript.compose(output)
    for figure in layout["figures"]:
        for extension in ("pdf", "svg", "png"):
            assert (output / "figures" / f"{figure}.{extension}").is_file()
    assert not (output / "panels").exists()
    manifest = json.loads((output / "figure_manifest.json").read_text())
    assert manifest["plot_data"] == str(data_path.relative_to(output))
    assert manifest["plot_data_sha256"] == sha(data_path)
    specs = json.loads(data_path.read_text())
    assert specs["panels"]["policy_p00"]["rows"][0]["delta_separation"] == 0
    assert specs["panels"]["case_fixture_separation"]["arms"]["hold"]["step"] == [128, 640]
    assert specs["panels"]["case_fixture_separation"]["threshold"] == 0.4


def _runs():
    return pd.DataFrame([
        {"task_id": task, "stage": "late", "arm": arm, "start_step": 128,
         "start_separation": 0.5, "start_excess": 0.05, "end_separation": value,
         "end_excess": 0.1, "setting": "a", "sensitivity": "core", "influence": 0.1,
         "rewiring": 0.01, "repost": 0.1, "replicate": int(task), "epsilon": 0.45,
         "history": 0, "bridge_shown": 0, "cross_confident": 0, "updates": 100}
        for task, arms in (("0", (("hold", 0.8), ("switch", 0.2))), ("1", (("hold", 0.8),)))
        for arm, value in arms
    ])


def test_pairs_exclude_unmatched_arms_and_reject_checkpoint_mismatch():
    runs = _runs()
    paired = paired_effects(runs, "switch", "hold")
    assert len(paired) == 1
    assert paired.iloc[0]["delta_separation"] == pytest.approx(-0.6)
    runs.loc[runs["arm"] == "switch", "start_step"] = 256
    with pytest.raises(ValueError, match="different start_step"):
        paired_effects(runs, "switch", "hold")


def test_pairs_reject_duplicates():
    runs = _runs()
    with pytest.raises(ValueError, match="duplicate"):
        paired_effects(pd.concat([runs, runs.iloc[:1]]), "switch", "hold")


def test_unsupported_layout_schema_is_rejected(tmp_path):
    atomic_json(tmp_path / "plot_data.json", {"schema": 99})
    atomic_json(tmp_path / "layout.json", {"schema": 1})
    with pytest.raises(ValueError, match="schema"):
        manuscript.compose(tmp_path)


def test_saved_panel_rows_reject_missing_measurement_fields():
    with pytest.raises(TypeError, match="task_id"):
        parse_panel({"kind": "bridge", "metric": "dose", "title": "fixture",
                     "rows": [{"bridge_shown": 5, "updates": 100}]})


def test_manifest_failure_keeps_previous_bundle_readable(tmp_path, monkeypatch):
    output = tmp_path / "bundle"
    (output / "data").mkdir(parents=True)
    (output / "data/old.txt").write_text("previous observations")
    atomic_json(output / "data/manifest.json", bundle_manifest())
    source = tmp_path / "source"
    source.mkdir()
    _binding(source)
    plan = freeze_plan(source / "plan.json", fixture_plan(tasks=generate_tasks("staged")[:1]))
    with DocumentStore(source / DATABASE):
        pass
    monkeypatch.setattr(shared_bundle, "read_study", lambda *args: (
        plan, {plan.tasks[0].id: staged()},
    ))

    def analyze_fixture(source, output, **kwargs):
        output.mkdir()
        return fixture_report(plan.plan_fingerprint)

    monkeypatch.setattr(shared_bundle, "analyze_study", analyze_fixture)
    write_json = shared_bundle.atomic_json

    def fail_manifest(path, content):
        if path.name == "manifest.json":
            raise OSError("injected manifest write failure")
        write_json(path, content)

    monkeypatch.setattr(shared_bundle, "atomic_json", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        shared_bundle.export_bundle(source, output)
    assert (output / "data/old.txt").read_text() == "previous observations"
    assert shared_bundle.read_bundle_manifest(output).schema == shared_bundle.SCHEMA
    assert not (output / ".previous-data").exists()


def test_shared_export_cli_accepts_study_data_without_a_manuscript_wrapper(tmp_path, monkeypatch, capsys):
    dataset, output = tmp_path / "dataset", tmp_path / "bundle"
    calls = []

    def export(source, destination, *, require_complete):
        calls.append((source, destination, require_complete))
        return bundle_manifest(profile="recommender_variants", completed=1)

    monkeypatch.setattr(shared_bundle, "export_bundle", export)
    shared_bundle.main(["export", "--dataset", str(dataset), "--output", str(output), "--require-complete"])
    assert calls == [(dataset, output, True)]
    assert json.loads(capsys.readouterr().out) == to_wire(bundle_manifest(profile="recommender_variants", completed=1))


def test_stage_manuscript_cli_rejects_the_removed_export_entry(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        manuscript.main(["export", "--dataset", str(tmp_path), "--output", str(tmp_path / "bundle")])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert not (tmp_path / "bundle").exists()


def test_variant_export_counts_complete_arms_and_keeps_previous_generation(tmp_path):
    source, bundle = tmp_path / "source", tmp_path / "bundle"
    _binding(source)
    task = variants_design.generate_tasks()[0]
    protocol = variants_design.design_protocol()
    plan = freeze_plan(source / "plan.json", fixture_plan("recommender_variants", tasks=(task,)))
    original = [arm for family in protocol.variants.values() for arm in family]
    record = FullResult(arms={arm: _arm(0) for arm in original})
    with StudyStore(source) as store:
        store.put_task(task, record, plan=plan)
    manifest = shared_bundle.export_bundle(source, bundle)
    assert manifest.profile == "recommender_variants"
    assert manifest.available_records == 1 and manifest.completed == 0
    previous = sha(bundle / "data/manifest.json")
    with pytest.raises(ValueError, match="incomplete: 0/1"):
        shared_bundle.export_bundle(source, bundle, require_complete=True)
    assert sha(bundle / "data/manifest.json") == previous
    record.arms.update({arm: _arm(0) for arm in variant_arms(protocol)})
    with StudyStore(source) as store:
        store.put_task(task, record, plan=plan)
    complete = shared_bundle.export_bundle(source, bundle, require_complete=True)
    assert complete.completed == complete.planned == 1


def test_export_counts_the_copied_generation_when_a_record_advances(tmp_path, monkeypatch):
    source, bundle = tmp_path / "source", tmp_path / "bundle"
    _binding(source)
    task = variants_design.generate_tasks()[0]
    protocol = variants_design.design_protocol()
    plan = freeze_plan(source / "plan.json", fixture_plan("recommender_variants", tasks=(task,)))
    original = [arm for family in protocol.variants.values() for arm in family]
    record = FullResult(arms={arm: _arm(0) for arm in original})
    with StudyStore(source) as store:
        store.put_task(task, record, plan=plan)
    backup = DocumentStore.backup

    def backup_then_advance(store, destination):
        result = backup(store, destination)
        record.arms.update({arm: _arm(0) for arm in variant_arms(protocol)})
        with StudyStore(source) as current:
            current.put_task(task, record, plan=plan)
        return result

    monkeypatch.setattr(DocumentStore, "backup", backup_then_advance)
    copied = shared_bundle.export_bundle(source, bundle)
    assert copied.completed == 0 and copied.planned == 1
    monkeypatch.setattr(DocumentStore, "backup", backup)
    manifest = shared_bundle.export_bundle(source, bundle, require_complete=True)
    assert manifest.completed == manifest.planned == 1


def test_interrupted_old_generation_rename_is_recovered(tmp_path):
    backup = tmp_path / ".previous-data"
    backup.mkdir()
    atomic_json(backup / "manifest.json", bundle_manifest())
    assert shared_bundle.read_bundle_manifest(tmp_path).schema == shared_bundle.SCHEMA
    assert (tmp_path / "data/manifest.json").is_file()
    assert not backup.exists()


def test_bridge_limits_expand_to_every_saved_effect():
    specs = _specs({
        "p": BridgePanel(title="p", metric="delta_separation", rows=(_effect(delta_separation=-1), _effect(delta_separation=1))),
        "e": BridgePanel(title="e", metric="delta_excess", rows=(_effect(delta_excess=.8),)),
    })
    limits = manuscript._bridge_limits(specs)
    assert limits["delta_separation"][0] < -1 and limits["delta_separation"][1] > 1
    assert limits["delta_excess"][1] > 0.8


def test_bridge_dose_panels_use_a_shared_scale():
    panels = {
        name: BridgePanel(metric="dose", title=name, rows=(_effect(
            repost=.1, replicate=0, bridge_shown=count, updates=1000),))
        for name, count in (("low", 40), ("high", 230))
    }
    specs = _specs(panels)
    style = manuscript._figure_style(specs, _layout(marker_size=16))
    fig, axes = plt.subplots(1, 2)
    try:
        for ax, panel in zip(axes, panels.values()):
            manuscript._draw_panel(fig, ax, panel, style)
        assert axes[0].get_ylim() == axes[1].get_ylim()
        assert axes[0].get_ylim()[0] < 0 < 230 < axes[0].get_ylim()[1]
        assert all(ax.lines[0].get_zorder() > ax.collections[0].get_zorder() for ax in axes)
    finally:
        plt.close(fig)


def test_case_panels_keep_negative_excess_and_omit_unrelated_arm():
    case = CasePanel(metric="excess_concordance", title="case", threshold=.1,
        arms={name: CaseCurve(step=(64, 576), value=tuple(values)) for name, values in (
                ("hold", [-.2, .4]), ("switch", [-.2, .55]), ("hide_reposts", [-.2, .3]))})
    style = manuscript._figure_style(_specs({"case": case}), _layout())
    fig, ax = plt.subplots()
    try:
        manuscript._draw_panel(fig, ax, case, style)
        assert ax.get_ylim()[0] < -0.2 and ax.get_ylim()[1] > 0.55
        assert [line.get_label() for line in ax.lines[:2]] == ["Keep OpinionM9", "Switch to StructureM9"]
        assert len(ax.lines) == 3  # Two displayed arms and the state threshold.
        assert ax.lines[0].get_zorder() > ax.lines[1].get_zorder()
    finally:
        plt.close(fig)


def test_swarm_preserves_values_and_separates_markers_in_display_space():
    import numpy as np

    values = np.array([0.0, 0.0, 0.002, -0.002, 0.0, 0.003])
    original = values.copy()
    fig, ax = plt.subplots(figsize=(2.2, 2.0), dpi=120)
    try:
        ax.set(xlim=(-0.5, 2.5), ylim=(-0.1, 0.1))
        fig.canvas.draw()
        positions, size = manuscript._swarm_positions(ax, 1.0, values, marker_size=9)
        repeated_positions, repeated_size = manuscript._swarm_positions(ax, 1.0, values, marker_size=9)
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(positions, repeated_positions)
        assert repeated_size == size
        assert len(positions) == len(values)
        assert np.all(np.abs(positions - 1.0) <= 0.29)
        pixels = ax.transData.transform(np.column_stack((positions, values)))
        distances = np.linalg.norm(pixels[:, None, :] - pixels[None, :, :], axis=2)
        assert distances[np.triu_indices(len(values), k=1)].min() >= np.sqrt(size) * fig.dpi / 72
        restored = ax.transData.inverted().transform(pixels)
        np.testing.assert_allclose(restored[:, 1], original, atol=1e-15)
    finally:
        plt.close(fig)


def test_bridge_collapses_zero_insertion_counts_but_keeps_inserted_repeats():
    import numpy as np
    from matplotlib.colors import to_rgba

    rows = [_effect(repost=0., replicate=replicate, bridge_shown=0,
             updates=1000, delta_separation=0.) for replicate in range(16)]
    rows.extend(_effect(repost=.1, replicate=replicate, bridge_shown=1,
                 updates=1000, delta_separation=value)
                for replicate, value in enumerate((-0.2, 0.1, 0.1)))
    panel = BridgePanel(metric="delta_separation", title="fixture", rows=tuple(rows))
    style = manuscript._figure_style(_specs({"fixture": panel}), _layout(marker_size=9))
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    try:
        manuscript._draw_panel(fig, ax, panel, style)
        assert len(ax.collections) == 2
        zero, inserted = ax.collections
        assert zero.get_offsets().shape == (1, 2)
        assert zero.get_offsets()[0, 1] == 0
        np.testing.assert_allclose(zero.get_facecolors()[0], to_rgba("white"))
        gray = zero.get_edgecolors()[0, :3]
        assert gray[0] == pytest.approx(gray[1]) == pytest.approx(gray[2])
        assert 0 < gray[0] < 1
        assert [text.get_text() for text in ax.texts].count("×16") == 1
        assert inserted.get_offsets().shape == (3, 2)
        np.testing.assert_allclose(sorted(inserted.get_offsets()[:, 1]), [-0.2, 0.1, 0.1])
        np.testing.assert_allclose(inserted.get_facecolors(), inserted.get_edgecolors())
        means = [line for line in ax.lines if to_rgba(line.get_color()) == to_rgba("black")]
        assert len(means) == 2
        np.testing.assert_allclose(means[0].get_ydata(), [0, 0])
        assert min(means[0].get_xdata()) > zero.get_offsets()[0, 0]
        np.testing.assert_allclose(means[1].get_ydata(), [0, 0], atol=1e-15)
        assert min(means[1].get_xdata()) > inserted.get_offsets()[:, 0].max()
    finally:
        plt.close(fig)


def test_case_overlap_symbols_keep_recorded_paths_and_report_last_shared_step():
    import numpy as np

    steps = [256 + 16 * index for index in range(33)]
    shared = [0.8] * len(steps)
    bridge = shared[:27] + [0.0] * 6
    panel = CasePanel(metric="separation", title="fixture", threshold=.4,
        arms={name: CaseCurve(step=tuple(steps), value=tuple(values)) for name, values in (
                 ("hold", shared), ("switch", shared), ("bridge", bridge))})
    style = manuscript._figure_style(_specs({"fixture": panel}), _layout())
    fig, ax = plt.subplots(figsize=(2.2, 2.0))
    try:
        manuscript._draw_panel(fig, ax, panel, style)
        curves = ax.lines[:3]
        # Matplotlib normalizes every custom dash sequence to the name "--".
        patterns = {(line.get_linestyle(), tuple(line._unscaled_dash_pattern[1] or ()))
                    for line in curves}
        assert len(patterns) == 3
        assert len({line.get_marker() for line in curves}) == 3
        marker_steps = []
        for line, arm in zip(curves, ("hold", "switch", "bridge")):
            np.testing.assert_array_equal(line.get_xdata(), steps)
            np.testing.assert_array_equal(line.get_ydata(), panel.arms[arm].value)
            start, interval = line.get_markevery()
            selected = set(range(start, len(steps), interval))
            assert 1 < len(selected) < len(steps) / 2
            marker_steps.append(selected)
        assert not (marker_steps[0] & marker_steps[1] or marker_steps[0] & marker_steps[2]
                    or marker_steps[1] & marker_steps[2])
        note = manuscript._case_overlap_note(panel)
        assert "672" in note and "688" not in note
        assert "recorded" in note.lower()
        assert note in [text.get_text() for text in ax.texts]
        middle = replace(panel, arms={arm: panel.arms[arm] for arm in ("hold", "switch")})
        assert manuscript._case_overlap_note(middle) is None
    finally:
        plt.close(fig)
