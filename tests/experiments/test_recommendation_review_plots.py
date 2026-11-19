"""Review figures retain per-repeat effects and their actual observation clock."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from experiments.paper.recommendation import plot, review_plots


def test_bridge_review_keeps_one_influential_repeat_visible(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(review_plots, "plt_save_and_close", lambda fig, path: figures.append(fig))
    rows = []
    for replicate, difference in enumerate((0, -0.892, 0)):
        for arm, value in (("switch", 0), ("bridge", difference)):
            rows.append({"task_id": str(replicate), "stage": "late", "influence": 0.1,
                         "rewiring": 0.01, "repost": 0.1, "arm": arm,
                         "epsilon": 0.45, "history": 0,
                         "separation_change": value, "excess_change": value})
    review_plots.plot_bridge_replicates(pd.DataFrame(rows), tmp_path, [(0.1, 0.01)])
    try:
        values = figures[0].axes[0].collections[0].get_offsets()[:, 1]
        np.testing.assert_allclose(values, [0, -0.892, 0])
        assert any(text.get_text() == "n=3" for text in figures[0].axes[0].texts)
    finally:
        for fig in figures:
            plt.close(fig)


def test_pathway_grid_retains_counts_and_unplanned_cells(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(review_plots, "plt_save_and_close", lambda fig, path: figures.append(fig))
    counts = pd.DataFrame([
        {"alpha": 0.1, "q": 0.01, "p": 0, "n": 3, "PbS": 3, "epsilon": 0.45, "history": 0},
        {"alpha": 0.3, "q": 0.03, "p": 0, "n": 3, "PbS": 2, "epsilon": 0.45, "history": 0},
    ])
    review_plots.plot_reference_pbs_grid(counts, tmp_path)
    try:
        matrix = figures[0].axes[0].images[0].get_array()
        assert matrix[0, 0] == 1 and np.ma.getmaskarray(matrix)[0, 1]
        assert {text.get_text() for text in figures[0].axes[0].texts} == {"3/3", "2/3", "—"}
    finally:
        plt.close(figures[0])


def test_bridge_review_rejects_mixed_measurements(tmp_path):
    rows = pd.DataFrame([
        {"stage": "late", "influence": 0.05, "rewiring": 0.05,
         "epsilon": epsilon, "history": 0}
        for epsilon in (0.35, 0.45)
    ])
    with pytest.raises(ValueError, match="one epsilon and history"):
        review_plots.plot_bridge_replicates(rows, tmp_path, [(0.05, 0.05)])


@pytest.mark.parametrize("invalid", ["duplicate", "zero", "outside", "mixed"])
def test_pathway_grid_rejects_ambiguous_or_invalid_counts(tmp_path, invalid):
    row = {"alpha": 0.1, "q": 0.01, "p": 0, "n": 3, "PbS": 3, "epsilon": 0.45, "history": 0}
    rows = [row]
    if invalid == "duplicate":
        rows.append(dict(row))
    elif invalid == "mixed":
        rows.append({**row, "alpha": 0.3, "history": 2})
    else:
        rows[0] = {**row, "n" if invalid == "zero" else "PbS": 0 if invalid == "zero" else 4}
    with pytest.raises(ValueError):
        review_plots.plot_reference_pbs_grid(pd.DataFrame(rows), tmp_path)


def test_case_plot_uses_absolute_steps_of_each_checkpoint(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(review_plots, "plt_save_and_close", lambda fig, path: figures.append(fig))
    record = {"stages": {phase: {
        "checkpoint": {"step": start}, "arms": {
            "hold": {"endpoints": [
                {"step": start, "pathway": {"separation": 0.6, "excess_concordance": 0.05}},
                {"step": start + 512, "pathway": {"separation": 0.9, "excess_concordance": 0.2}},
            ]},
        },
    } for phase, start in (("middle", 64), ("late", 256))}}
    review_plots.plot_intervention_case(record, tmp_path, title="one initial state")
    try:
        np.testing.assert_array_equal(figures[0].axes[0].lines[0].get_xdata(), [64, 576])
        np.testing.assert_array_equal(figures[0].axes[1].lines[0].get_xdata(), [256, 768])
    finally:
        plt.close(figures[0])


def test_tables_only_cli_disables_full_parameter_figure_rendering(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(plot, "analyze", lambda source, output, **options: calls.append((source, output, options)))
    plot.main([str(tmp_path / "source"), "--output", str(tmp_path / "tables"), "--tables-only"])
    assert calls == [(tmp_path / "source", tmp_path / "tables", {"make_figures": False})]
