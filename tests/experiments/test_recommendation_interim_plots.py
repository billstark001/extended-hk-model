"""Interim figures keep parameter coverage and actual bridge exposure explicit."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from experiments.paper.recommendation import plot
from experiments.paper.recommendation_shared import plot as shared_plot
from experiments.paper.recommendation_shared.analysis import EFFECTS


def test_coverage_distinguishes_unplanned_and_unfinished_cells(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(shared_plot, "plt_save_and_close", lambda fig, path: captured.append(fig))
    coverage = pd.DataFrame([
        {"setting": "a", "influence": 0.01, "rewiring": 0.3, "epsilon": 0.45, "history": 0,
         "repost": p, "planned": 16, "completed": n}
        for p, n in ((0, 16), (0.1, 0))
    ] + [
        {"setting": "b", "influence": 0.05, "rewiring": 0.05, "epsilon": 0.35, "history": 0,
         "repost": 0.1, "planned": 16, "completed": 2}
    ])
    shared_plot.plot_coverage(coverage, tmp_path)
    fig = captured[0]
    try:
        matrix = fig.axes[0].images[0].get_array()
        assert matrix[0, 0] == 1 and matrix[0, 1] == 0
        assert np.ma.getmaskarray(matrix)[1, 0]
        assert {text.get_text() for text in fig.axes[0].texts} == {"16/16", "0/16", "2/16", "—"}
    finally:
        plt.close(fig)


def test_exposure_plot_includes_inserted_bridge_slots_and_all_slot_total(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(plot, "plt_save_and_close", lambda fig, path: captured.append(fig))
    rows = pd.DataFrame([{
        "stage": "late", "arm": "bridge", "epsilon": 0.45, "history": 0, "repost": 0.1,
        "updates": 100,
        "non_exploration_cross_camp": 500, "non_exploration_cross_confident": 300,
        "non_exploration_cross_rewire": 100,
        "cross_camp": 700, "cross_confident": 400, "cross_rewire": 120,
        "bridge_cross_camp": 490, "bridge_cross_confident": 290, "bridge_cross_rewire": 90,
    }])
    plot.plot_late_exposure(rows, tmp_path)
    plot.plot_late_exposure(rows, tmp_path, all_slots=True)
    try:
        np.testing.assert_array_equal(captured[0].axes[0].lines[0].get_ydata(), [5, 3, 1])
        np.testing.assert_array_equal(captured[0].axes[0].lines[1].get_ydata(), [4.9, 2.9, 0.9])
        np.testing.assert_array_equal(captured[1].axes[0].lines[0].get_ydata(), [7, 4, 1.2])
        assert "Policy + bridge" in captured[0].axes[0].get_ylabel()
        assert "All recommendation" in captured[1].axes[0].get_ylabel()
    finally:
        for fig in captured:
            plt.close(fig)


def test_effects_select_one_parameter_cell_without_pooling(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(shared_plot, "plt_save_and_close", lambda fig, path: captured.append(fig))
    contrasts = pd.DataFrame([
        {"setting": setting, "repost": repost, "epsilon": 0.45, "history": 0, "stage": "late",
         "arm": "switch", "metric": metric, "n": 16, "mean": mean, "lo": mean - 0.01, "hi": mean + 0.01}
        for setting, repost, mean in (("all", "all", 0.4), ("a", 0.1, -0.2), ("b", 0.1, 0.8))
        for metric in shared_plot.METRICS
    ])
    shared_plot.plot_effects(contrasts, tmp_path, ("late",), stage_axis=True, setting="a", repost=0.1)
    fig = captured[0]
    try:
        assert float(fig.axes[0].lines[1].get_xdata()[0]) == pytest.approx(-0.2)
    finally:
        plt.close(fig)


def test_interim_analyze_pools_only_complete_cells_and_keeps_source_hashes(tmp_path, monkeypatch):
    directory = tmp_path / "source"
    (directory / "records").mkdir(parents=True)
    tasks, rows = [], []
    for identifier, setting in (("a0", "a"), ("a1", "a"), ("b0", "b"), ("b1", "b")):
        tasks.append({"id": identifier, "setting": setting, "sensitivity": "core", "replicate": int(identifier[-1]),
                      "metadata": {"HKParams": {"Tolerance": 0.45, "Influence": 0.01 if setting == "a" else 0.03,
                                                "RewiringRate": 0.3, "RepostRate": 0.1}, "PostRetainCount": 0}})
        if identifier == "b1":
            continue
        (directory / "records" / (identifier + ".json.gz")).write_bytes(identifier.encode())
        for arm in ("hold", "switch"):
            rows.append({"task_id": identifier, "setting": setting, "sensitivity": "core", "epsilon": 0.45,
                         "history": 0, "repost": 0.1, "stage": "late", "arm": arm,
                         **dict.fromkeys(EFFECTS, 0.0)})
    plan = {"tasks": tasks, "profile": "staged", "plan_fingerprint": "fixture"}
    records = {identifier: {"stages": {"late": {}}} for identifier in ("a0", "a1", "b0")}
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(plot, "read_study", lambda directory: (plan, records))
    monkeypatch.setattr(plot, "study_tables", lambda plan, records, **kwargs: (frame, frame))
    monkeypatch.setattr(plot, "reference_tables", lambda plan, records: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(shared_plot, "plot_coverage", lambda *args: None)
    monkeypatch.setattr(plot, "plot_late_opinions", lambda *args, **kwargs: None)
    scopes = []
    monkeypatch.setattr(plot, "_plot_scope", lambda study, sample, *args, **kwargs:
                        scopes.append((set(sample["task_id"]), kwargs["title"])))
    report = plot.analyze(directory, tmp_path / "analysis")
    assert report["completed"] == 3 and report["planned"] == 4 and report["complete"] is False
    assert report["pooled_figures"] == {"rule": "completed_parameter_cells_only", "tasks": 2, "cells": 1}
    assert report["stage_gate"]["status"] == "pending"
    assert set(report["record_sha256"]) == {"a0", "a1", "b0"}
    assert scopes[-1][0] == {"a0", "a1"} and "1 complete cells, 2 tasks" in scopes[-1][1]
    assert any(ids == {"b0"} and "1/2 tasks" in title for ids, title in scopes)
    persisted = json.loads((tmp_path / "analysis" / "coverage.json").read_text())
    assert persisted["record_sha256"] == report["record_sha256"]


def test_late_opinion_heatmap_keeps_replicates_and_three_modes_separate(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(plot, "plt_save_and_close", lambda fig, path: captured.append(fig))
    tasks = [{"id": "second", "replicate": 2}, {"id": "first", "replicate": 1}]
    records = {
        task["id"]: {"stages": {"late": {"arms": {
            "hold": {"final_opinions": [-0.8] * 6 + [0.8] * 6},
            "bridge": {"final_opinions": [-0.8] * 4 + [0.0] * 4 + [0.8] * 4},
        }}}}
        for task in tasks
    }
    plot.plot_late_opinions(tasks, records, tmp_path, title="fixture")
    fig = captured[0]
    try:
        matrices = [ax.images[0].get_array() for ax in fig.axes[:2]]
        assert all(matrix.shape == (2, 100) for matrix in matrices)
        assert np.count_nonzero(matrices[0][0]) == 2
        assert np.count_nonzero(matrices[1][0]) == 3
        np.testing.assert_array_almost_equal(matrices[1].sum(axis=1), [1, 1])
        assert fig.axes[0].images[0].get_clim() == fig.axes[1].images[0].get_clim()
        assert [text.get_text() for text in fig.axes[0].get_yticklabels()] == ["1", "2"]
    finally:
        plt.close(fig)
