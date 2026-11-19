"""Synthetic checks for state classification and paired recommendation analysis."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ehk.metrics.homophily import mean_followee_edge_concordance, uniform_concordance_probability
from ehk.metrics.pathway import first_passage
from ehk.micro.observables import get_gradation_index_hp
from experiments.paper.recommendation.analysis import (
    EFFECTS, contrast_table, coverage_by_setting, exposure_columns, stage_gate, stage_interactions,
)
from experiments.paper.recommendation import design as cfg
from experiments.paper.recommendation.design import generate_tasks, select_stages


def test_consensus_and_separated_opinions_have_different_passages():
    assert uniform_concordance_probability(0.45) == pytest.approx(0.399375)
    assert first_passage([(0, 1, 0.5)]).label == "simultaneous"
    assert first_passage([(0, 0, 0), (16, 1, 0), (32, 1, 0.2)]).label == "PbS"
    assert first_passage([(0, 0, 0), (16, 0, 0.2), (32, 1, 0.2)]).label == "SbP"
    assert first_passage([(0, 0, 0), (16, 1, 0)]).label == "separation_only"
    assert first_passage([(0, 0, 0), (16, 0, 0.2)]).label == "segregation_only"
    assert first_passage([(0, 0, None), (16, 0, 0)]).label == "unavailable"
    assert first_passage([(0, 0, 0), (16, 0, 0)]).label == "neither"


def test_edge_and_agent_weighting_are_explicit_and_iw_is_signed():
    graph = nx.DiGraph([(0, 1), (0, 2), (1, 0)])
    assert mean_followee_edge_concordance(graph, np.array([0, 0, 1]), 0.45) == pytest.approx(0.75)
    p, h = np.array([0, 0.5, 0.1]), np.array([0, 0, 1])
    assert get_gradation_index_hp(p, h) == pytest.approx(-0.2)


def test_checkpoints_use_only_current_state_and_recent_activity():
    def point(step, separation, excess, move):
        return {
            "step": step,
            "pathway": {"separation": separation, "excess_concordance": excess},
            "exposure": {"updates": int(step > 0) * 10, "rewires": 0, "absolute_shift": move},
        }

    points = [point(0, 0.23, 0, 0), point(16, 0.42, 0.02, 1), point(112, 0.5, 0.09, 0), point(128, 0.5, 0.09, 0)]
    assert {stage: row["step"] for stage, row in select_stages(points).items()} == {
        "early": 0,
        "middle": 16,
        "late": 128,
    }
    assert set(select_stages(points[:-1])) == {"early", "middle"}


def test_timing_gate_uses_same_seed_stage_contrasts_and_complete_evidence():
    def stage(effect):
        return {
            "arms": {
                name: {"endpoints": [{"pathway": {"separation": value}}]}
                for name, value in {"hold": 0.25, "switch": 0.25 + effect}.items()
            }
        }

    tasks = [{"id": str(i), "sensitivity": "core"} for i in range(4)]
    records = {task["id"]: {"stages": {"early": stage(0), "late": stage(0.1)}} for task in tasks}
    assert stage_gate(tasks, records)["passed"]
    sensitivity = [{"id": f"s{i}", "sensitivity": "epsilon"} for i in range(2)]
    records.update({task["id"]: {"stages": {"early": stage(0), "late": stage(-0.2)}} for task in sensitivity})
    gate = stage_gate([*tasks, *sensitivity], records)
    assert gate["passed"] and gate["n"] == gate["planned_core"] == gate["completed_core"] == 4
    records.pop("0")
    assert not stage_gate([*tasks, *sensitivity], records)["passed"]


def test_design_covers_core_cells_and_paired_single_factor_sensitivities():
    staged, timing, variants = (generate_tasks(name) for name in ("staged", "staged_timing", "recommender_variants"))
    assert len(staged) == len(variants) == 304 and len(timing) == 240
    assert cfg.SETTINGS == [(0.01, 0.30), (0.03, 0.10), (0.05, 0.05), (0.10, 0.03), (0.30, 0.01)]
    assert all(task["sensitivity"] == "core" for task in timing)
    for a, b in zip(staged, timing):
        assert a["metadata"]["RNG"] != b["metadata"]["RNG"]
    for tasks in (staged, variants):
        assert len({task["id"] for task in tasks}) == len(tasks)
        cells = {}
        for task in tasks:
            metadata = task["metadata"]
            key = task["setting"], metadata["HKParams"]["RepostRate"]
            cells.setdefault(key, set()).add(task["replicate"])
        assert len(cells) == 19 and all(replicates == set(range(16)) for replicates in cells.values())
        center = {
            task["replicate"]: task["metadata"]["RNG"]
            for task in tasks if task["sensitivity"] == "core"
            and task["metadata"]["HKParams"]["Influence"] == 0.05
            and task["metadata"]["HKParams"]["RepostRate"] == 0.1
        }
        for task in tasks:
            if task["sensitivity"] != "core":
                assert task["metadata"]["RNG"] == center[task["replicate"]]


def test_contrasts_and_stage_interactions_keep_measurement_settings_separate():
    rows = []
    for group, epsilon, history, delta in (("core", 0.45, 0, 0.1), ("epsilon", 0.35, 0, 0.5), ("history", 0.45, 2, -0.1)):
        for replicate in range(2):
            for stage in ("early", "late"):
                for arm in ("hold", "switch"):
                    row = {
                        "task_id": f"{group}-{replicate}", "sensitivity": group, "setting": group,
                        "epsilon": epsilon, "history": history, "repost": 0.1, "stage": stage, "arm": arm,
                        **dict.fromkeys(EFFECTS, 0.0),
                    }
                    row["separation_change"] = delta if stage == "late" and arm == "switch" else 0
                    rows.append(row)
    runs = pd.DataFrame(rows)
    contrasts = contrast_table(runs, "staged")
    pooled = contrasts[(contrasts["setting"] == "all") & (contrasts["stage"] == "late") & (contrasts["metric"] == "separation_change")]
    assert len(pooled) == 3 and set(pooled["n"]) == {2}
    assert dict(zip(pooled["sensitivity"], pooled["mean"])) == pytest.approx({"core": 0.1, "epsilon": 0.5, "history": -0.1})
    interactions = stage_interactions(runs)
    pooled = interactions[(interactions["setting"] == "all") & (interactions["metric"] == "separation_change") & (interactions["earlier"] == "early") & (interactions["later"] == "late")]
    assert dict(zip(pooled["sensitivity"], pooled["mean"])) == pytest.approx({"core": 0.1, "epsilon": 0.5, "history": -0.1})


def test_setting_coverage_keeps_missing_stages_visible():
    tasks = generate_tasks("staged")
    records = {tasks[0]["id"]: {"stages": {"early": {}}}}
    coverage = coverage_by_setting({"profile": "staged", "tasks": tasks}, records)
    assert len(coverage) == 19 and sum(row["planned"] for row in coverage) == 304
    assert sum(row["completed"] for row in coverage) == 1
    assert sum(row["stages"]["early"] for row in coverage) == 1
    assert sum(row["stages"]["late"] for row in coverage) == 0


def test_new_exposure_protocol_requires_actual_counters():
    with pytest.raises(KeyError, match="requested"):
        exposure_columns({"updates": 5})


def test_new_analysis_rejects_old_design_schema(tmp_path, monkeypatch):
    from experiments.paper.recommendation import analysis

    monkeypatch.setattr(analysis, "read_plan", lambda path: {"study": "paper_recommendation", "profile": "staged", "schema": 1})
    with pytest.raises(ValueError, match="unsupported recommendation design version"):
        analysis.read_study(tmp_path)


def test_late_exposure_plot_covers_every_repost_setting(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from experiments.paper.recommendation import plot

    captured = []

    def save(fig, path):
        captured.append([ax.get_title() for ax in fig.axes])
        plt.close(fig)

    monkeypatch.setattr(plot, "plt_save_and_close", save)
    rows = pd.DataFrame([
        {"stage": "late", "arm": "hold", "epsilon": 0.45, "history": 0, "repost": repost,
         "updates": 10, "policy_cross_camp": 2, "policy_cross_confident": 1, "policy_cross_rewire": 0}
        for repost in cfg.REPOST_RATES
    ])
    plot.plot_late_exposure(rows, tmp_path)
    assert captured == [["Repost probability 0", "Repost probability 0.1", "Repost probability 0.25"]]
