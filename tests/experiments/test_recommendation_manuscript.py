"""Portable figure stages preserve paired units and require no raw-data paths."""

import json
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from common.experiments.plans import freeze_plan
from common.io.json import atomic_json, fingerprint, save_result, sha
from experiments.paper.recommendation import manuscript
from experiments.paper.recommendation.analysis import EXPOSURE_COUNTS, SLOT_MEASURES, SLOT_SOURCES
from experiments.paper.recommendation.design import DESIGN_VERSION, generate_tasks


def _point(step, start):
    exposure = dict.fromkeys(EXPOSURE_COUNTS, 0)
    exposure.update({source: dict.fromkeys(SLOT_MEASURES, 0) for source in SLOT_SOURCES})
    exposure["updates"] = 0 if step == start else 100
    return {"step": step, "relative_step": step - start, "exposure": exposure,
            "pathway": {"separation": 0.5, "excess_concordance": 0.05,
                        "edge_concordance": 0.55, "pair_concordance": 0.5}}


def _arm(start):
    return {"endpoints": [_point(start, start), _point(start + 512, start)],
            "terminal": {"status": "nonterminal"}, "final_opinions": [-0.8, -0.7, 0.7, 0.8],
            "raw_record": "/unmounted/test/raw", "stopped_at": start + 512}


@pytest.mark.parametrize("legacy_cache", [False, True])
def test_export_then_compose_without_source_bundle_or_panel_images(tmp_path, monkeypatch, legacy_cache):
    source, bundle, output = (tmp_path / name for name in ("source", "bundle", "paper"))
    source.mkdir()
    task = generate_tasks("staged")[0]
    freeze_plan(source / "plan.json", {"schema": DESIGN_VERSION, "study": "paper_recommendation",
                                       "profile": "staged", "tasks": [task],
                                       "protocol": {"separation_threshold": 0.4, "excess_threshold": 0.1}})
    record = {"reference": _arm(0), "stages": {}}
    for phase, start in (("early", 0), ("middle", 16), ("late", 128)):
        record["stages"][phase] = {"checkpoint": _point(start, start),
                                  "arms": {name: _arm(start) for name in ("hold", "switch", "bridge")}}
    save_result(source / "records" / f"{task['id']}.json.gz",
                {"task_fingerprint": fingerprint(task), "result": record})
    manifest = manuscript.export_bundle(source, bundle)
    assert manifest["completed"] == manifest["planned"] == 1
    shutil.rmtree(source)
    monkeypatch.setattr(manuscript, "CASES", (("fixture", task["id"], "late", "fixture"),))
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
    if legacy_cache:
        (output / "panels").mkdir()
        data_path = output / "panels/specs.json"
        (output / "plot_data.json").rename(data_path)
    manuscript.compose(output)
    for figure in layout["figures"]:
        for extension in ("pdf", "svg", "png"):
            assert (output / "figures" / f"{figure}.{extension}").is_file()
    if not legacy_cache:
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
    paired = manuscript.paired_effects(runs, "switch", "hold")
    assert len(paired) == 1
    assert paired.iloc[0]["delta_separation"] == pytest.approx(-0.6)
    runs.loc[runs["arm"] == "switch", "start_step"] = 256
    with pytest.raises(ValueError, match="different start_step"):
        manuscript.paired_effects(runs, "switch", "hold")


def test_pairs_reject_duplicates():
    runs = _runs()
    with pytest.raises(ValueError, match="duplicate"):
        manuscript.paired_effects(pd.concat([runs, runs.iloc[:1]]), "switch", "hold")


def test_unsupported_layout_schema_is_rejected(tmp_path):
    atomic_json(tmp_path / "panels/specs.json", {"schema": 99})
    atomic_json(tmp_path / "layout.json", {"schema": 1})
    with pytest.raises(ValueError, match="schema"):
        manuscript.compose(tmp_path)


def test_manifest_failure_keeps_previous_bundle_readable(tmp_path, monkeypatch):
    output = tmp_path / "bundle"
    (output / "data").mkdir(parents=True)
    (output / "data/old.txt").write_text("previous observations")
    atomic_json(output / "data/manifest.json", {"schema": 1, "files": {}})
    source = tmp_path / "source"
    source.mkdir()
    (source / "plan.json").write_text("{}")
    (source / "records").mkdir()
    (source / "records/task.json.gz").write_bytes(b"new observations")
    monkeypatch.setattr(manuscript, "read_study", lambda *args: ({"plan_fingerprint": "new", "tasks": [{}]}, {"task": {}}))
    monkeypatch.setattr(manuscript, "analyze", lambda source, output, **kwargs: output.mkdir())
    write_json = manuscript.atomic_json

    def fail_manifest(path, content):
        if path.name == "manifest.json":
            raise OSError("injected manifest write failure")
        write_json(path, content)

    monkeypatch.setattr(manuscript, "atomic_json", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        manuscript.export_bundle(source, output)
    assert (output / "data/old.txt").read_text() == "previous observations"
    assert manuscript._bundle_manifest(output)["schema"] == 1
    assert not (output / ".previous-data").exists()


def test_interrupted_old_generation_rename_is_recovered(tmp_path):
    backup = tmp_path / ".previous-data"
    backup.mkdir()
    atomic_json(backup / "manifest.json", {"schema": 1, "files": {}})
    assert manuscript._bundle_manifest(tmp_path)["schema"] == 1
    assert (tmp_path / "data/manifest.json").is_file()
    assert not backup.exists()


def test_bridge_limits_expand_to_every_saved_effect():
    specs = {"panels": {
        "p": {"kind": "bridge", "metric": "delta_separation", "rows": [{"delta_separation": -1}, {"delta_separation": 1}]},
        "e": {"kind": "bridge", "metric": "delta_excess", "rows": [{"delta_excess": 0.8}]},
    }}
    limits = manuscript._bridge_limits(specs)
    assert limits["delta_separation"][0] < -1 and limits["delta_separation"][1] > 1
    assert limits["delta_excess"][1] > 0.8


def test_bridge_dose_panels_use_a_shared_scale():
    panels = {
        name: {"kind": "bridge", "metric": "dose", "title": name, "rows": [{
            "repost": 0.1, "replicate": 0, "bridge_shown": count, "updates": 1000,
        }]}
        for name, count in (("low", 40), ("high", 230))
    }
    specs = {"panels": panels}
    style = manuscript._figure_style(specs, {"style": {"marker_size": 16}})
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
    case = {"kind": "case", "metric": "excess_concordance", "title": "case", "threshold": 0.1,
            "arms": {name: {"step": [64, 576], "value": values} for name, values in (
                ("hold", [-0.2, 0.4]), ("switch", [-0.2, 0.55]), ("hide_reposts", [-0.2, 0.3]))}}
    style = manuscript._figure_style({"panels": {"case": case}}, {"style": {"colors": manuscript.COLORS}})
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

    rows = [{"repost": 0.0, "replicate": replicate, "bridge_shown": 0,
             "updates": 1000, "delta_separation": 0.0} for replicate in range(16)]
    rows.extend({"repost": 0.1, "replicate": replicate, "bridge_shown": 1,
                 "updates": 1000, "delta_separation": value}
                for replicate, value in enumerate((-0.2, 0.1, 0.1)))
    panel = {"kind": "bridge", "metric": "delta_separation", "title": "fixture", "rows": rows}
    style = manuscript._figure_style({"panels": {"fixture": panel}}, {"style": {"marker_size": 9}})
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
    panel = {"kind": "case", "metric": "separation", "title": "fixture", "threshold": 0.4,
             "arms": {name: {"step": steps, "value": values} for name, values in (
                 ("hold", shared), ("switch", shared), ("bridge", bridge))}}
    style = manuscript._figure_style({"panels": {"fixture": panel}},
                                     {"style": {"colors": manuscript.COLORS}})
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
            np.testing.assert_array_equal(line.get_ydata(), panel["arms"][arm]["value"])
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
        middle = {**panel, "arms": {arm: panel["arms"][arm] for arm in ("hold", "switch")}}
        assert manuscript._case_overlap_note(middle) is None
    finally:
        plt.close(fig)
