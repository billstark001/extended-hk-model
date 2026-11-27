"""Synthetic checks for state classification and paired recommendation analysis."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from common.io.json import fingerprint
from ehk.metrics.homophily import mean_followee_edge_concordance, uniform_concordance_probability
from ehk.metrics.pathway import first_passage
from ehk.micro.observables import get_gradation_index_hp
from experiments.paper.recommendation import design as cfg
from experiments.paper.recommendation.analysis import (
    completed_cell_task_ids,
    contrast_table,
    coverage_by_setting,
    stage_gate,
    stage_interactions,
)
from experiments.paper.recommendation.design import generate_tasks, select_stages
from experiments.paper.recommendation_shared.analysis import (
    EFFECTS,
    EXPOSURE_COUNTS,
    SLOT_MEASURES,
    SLOT_SOURCES,
    exposure_columns,
    summarize_arm,
)
from experiments.paper.recommender_variants import design as variants_design


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
    assert get_gradation_index_hp(p, h) == pytest.approx(0.7)
    assert get_gradation_index_hp(p, h, endpoint_complement=False) == pytest.approx(-0.2)


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
    staged, timing = (generate_tasks(name) for name in ("staged", "staged_timing"))
    variants = variants_design.generate_tasks()
    assert len(staged) == len(variants) == 496 and len(timing) == 432
    assert cfg.SETTINGS == [
        (0.01, 0.30), (0.03, 0.10), (0.05, 0.05), (0.10, 0.03), (0.30, 0.01),
        (0.01, 0.10), (0.03, 0.30), (0.10, 0.01), (0.30, 0.03),
    ]
    assert all(task["sensitivity"] == "core" for task in timing)
    for a, b in zip(staged, timing):
        assert a["metadata"]["RNG"] != b["metadata"]["RNG"]
    for tasks in (staged, variants):
        assert len({task["id"] for task in tasks}) == len(tasks)
        assert [task["replicate"] for task in tasks] == sorted(task["replicate"] for task in tasks)
        cells = {}
        for task in tasks:
            metadata = task["metadata"]
            key = task["setting"], metadata["HKParams"]["RepostRate"]
            cells.setdefault(key, set()).add(task["replicate"])
        assert len(cells) == 31 and all(replicates == set(range(16)) for replicates in cells.values())
        center = {
            task["replicate"]: task["metadata"]["RNG"]
            for task in tasks if task["sensitivity"] == "core"
            and task["metadata"]["HKParams"]["Influence"] == 0.05
            and task["metadata"]["HKParams"]["RepostRate"] == 0.1
        }
        for task in tasks:
            if task["sensitivity"] != "core":
                assert task["metadata"]["RNG"] == center[task["replicate"]]


@pytest.mark.parametrize("study, expected_count, expected_hash", [
    ("staged", 304, "556698b6805ec4ca177dcb1f6f5a613caef9aa150bfe551c9ee10538a7e1c8d5"),
    ("staged_timing", 240, "f98a184b8f0ef72a6914c99a0a0298c60b8972e7aec1ce9c54f7a1f1976a9181"),
    ("recommender_variants", 304, "e6ded01aad20a8c0661f6deef4ea8e0f2a70fdb5f449bfd21233aad46b4e7ded"),
])
def test_extension_preserves_every_existing_task_field(study, expected_count, expected_hash):
    existing_settings = {(0.01, 0.30), (0.03, 0.10), (0.05, 0.05), (0.10, 0.03), (0.30, 0.01)}
    tasks = variants_design.generate_tasks() if study == "recommender_variants" else generate_tasks(study)
    existing = [task for task in tasks if (
        task["metadata"]["HKParams"]["Influence"], task["metadata"]["HKParams"]["RewiringRate"]
    ) in existing_settings]
    assert len(existing) == expected_count
    assert fingerprint(sorted(existing, key=lambda task: task["id"])) == expected_hash
    seeds = [fingerprint(task["metadata"]["RNG"]) for task in tasks if task["sensitivity"] == "core"]
    assert len(seeds) == len(set(seeds))


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
    assert len(coverage) == 31 and sum(row["planned"] for row in coverage) == 496
    assert sum(row["completed"] for row in coverage) == 1
    assert sum(row["stages"]["early"] for row in coverage) == 1
    assert sum(row["stages"]["late"] for row in coverage) == 0


def test_complete_cell_selection_excludes_partially_finished_repeats():
    tasks = generate_tasks("staged")
    first = [task for task in tasks if task["setting"] == tasks[0]["setting"]
             and task["metadata"]["HKParams"]["RepostRate"] == 0 and task["replicate"] < 2]
    second = [task for task in tasks if task["setting"] == tasks[0]["setting"]
              and task["metadata"]["HKParams"]["RepostRate"] == 0.1 and task["replicate"] < 2]
    selected = [*first, *second]
    records = {task["id"]: {"stages": {}} for task in selected[:3]}
    assert completed_cell_task_ids({"profile": "staged", "tasks": selected}, records) == {task["id"] for task in first}


def test_stage_passage_keeps_reference_order_and_marks_window_censoring():
    def point(step, separation, excess):
        exposure = dict.fromkeys(EXPOSURE_COUNTS, 0)
        exposure.update({source: dict.fromkeys(SLOT_MEASURES, 0) for source in SLOT_SOURCES})
        return {"step": step, "pathway": {"separation": separation, "excess_concordance": excess,
                                           "edge_concordance": 0.9, "pair_concordance": 0.7}, "exposure": exposure}

    prefix = [point(0, 0.1, 0), point(16, 0.1, 0.2)]
    arm = {"endpoints": [point(256, 0.6, 0.2), point(768, 0.7, 0.2)],
           "final_opinions": [-0.7, -0.7, 0.7, 0.7], "terminal": {"status": "unresolved"}, "raw_record": "raw/test"}
    arm["endpoints"][-1]["exposure"]["policy"]["cross_confident"] = 3
    arm["endpoints"][-1]["exposure"]["bridge"]["cross_confident"] = 7
    result = summarize_arm(arm, iter(prefix))
    assert result["pathway"] == "SbP"
    assert (result["segregation_step"], result["separation_step"]) == (16, 256)
    assert result["window_pathway"] == "left_censored"
    assert result["window_separation_step"] is None and result["window_segregation_step"] is None
    assert result["non_exploration_cross_confident"] == 10


def test_new_exposure_protocol_requires_actual_counters():
    with pytest.raises(KeyError, match="requested"):
        exposure_columns({"updates": 5})


def test_new_analysis_rejects_old_design_schema(tmp_path):
    from common.experiments.plans import freeze_plan
    from experiments.paper.recommendation import analysis

    freeze_plan(tmp_path / "plan.json", {
        "study": "paper_recommendation", "profile": "staged", "schema": 1,
        "tasks": [{"id": "old-design-fixture"}],
    })
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
         "updates": 10, "non_exploration_cross_camp": 2, "non_exploration_cross_confident": 1, "non_exploration_cross_rewire": 0,
         "cross_camp": 2, "cross_confident": 1, "cross_rewire": 0}
        for repost in cfg.REPOST_RATES
    ])
    plot.plot_late_exposure(rows, tmp_path)
    assert captured == [["Repost probability 0", "Repost probability 0.1", "Repost probability 0.25"]]
