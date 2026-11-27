"""Factorial additions retain existing observations and count only paired evidence."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from dataclasses import replace
from ehk.domain import Binding, FullResult
from tests.experiments.recommendation_fixtures import arm as fixture_arm, point, plan as fixture_plan

from experiments.paper.recommendation_shared import analysis as shared_analysis
from experiments.paper.recommendation_shared import plot
from experiments.paper.recommender_variants import analysis
from experiments.paper.recommender_variants import design as cfg


def _plan():
    return fixture_plan("recommender_variants", tasks=cfg.generate_tasks()[:2])


def _rows(plan, *, omit=None):
    values = (
        {(1, 0): 1, (4, 0): 3, (1, 2): 2, (4, 2): 8},
        {(1, 0): 2, (4, 0): 5, (1, 2): 4, (4, 2): 11},
    )
    rows = []
    for task, observed in zip(plan.tasks, values):
        for arm, configuration in cfg.variant_arms(plan.protocol).items():
            parameters = task.metadata.RecSysParams.with_overrides(configuration.parameters)
            factors = parameters.Steepness, parameters.OpRandomNoiseStd
            if configuration.recommender != "OpinionRandom" or (task.id, factors) == omit:
                continue
            rows.append({
                "task_id": task.id, "setting": task.setting, "sensitivity": "core",
                "epsilon": 0.45, "history": 0, "repost": task.metadata.HKParams.RepostRate,
                "stage": "full", "arm": arm, "recommender": "OpinionRandom",
                "zeta": factors[0], "score_noise": factors[1],
                **dict.fromkeys(shared_analysis.EFFECTS, observed[factors]),
            })
    return pd.DataFrame(rows).sample(frac=1, random_state=3)


def test_frozen_six_arm_plan_remains_complete_after_factorial_extension():
    plan = _plan()
    original = [arm for family in plan.protocol.variants.values() for arm in family]
    records = {task.id: FullResult(arms={name: fixture_arm() for name in original}) for task in plan.tasks}
    assert analysis.completed_task_ids(plan, records) == set()
    coverage = analysis.variant_arm_coverage(plan, records).groupby("arm")["completed"].sum()
    assert all(coverage[arm] == 2 for arm in original)
    assert sum(coverage[arm] for arm in coverage.index if arm not in original) == 0
    assert all(row.completed == 0 for row in analysis.coverage_by_setting(plan, records))
    plan.protocol = replace(plan.protocol, variant_arms=None)
    assert analysis.completed_task_ids(plan, records) == set(records)
    assert all(row.completed == row.planned for row in analysis.coverage_by_setting(plan, records))


def test_factorial_contrasts_pair_initial_states_and_difference_of_differences():
    plan = _plan()
    result = analysis.opinion_random_factorial(plan, _rows(plan))
    pooled = result[(result["setting"] == "all") & (result["metric"] == "separation_change")].set_index("contrast")
    expected = {
        "zeta_at_score_noise_0": 2.5, "zeta_at_score_noise_2": 6.5,
        "score_noise_at_zeta_1": 1.5, "score_noise_at_zeta_4": 5.5, "interaction": 4,
    }
    for name, mean in expected.items():
        assert pooled.loc[name, "n"] == 2
        assert pooled.loc[name, "mean"] == pytest.approx(mean)


def test_partial_factorial_uses_available_pairs_and_requires_all_four_for_interaction():
    plan = _plan()
    result = analysis.opinion_random_factorial(plan, _rows(plan, omit=(plan.tasks[1].id, (1, 0))))
    pooled = result[(result["setting"] == "all") & (result["metric"] == "separation_change")].set_index("contrast")
    assert pooled.loc["zeta_at_score_noise_2", "n"] == 2
    assert pooled.loc["score_noise_at_zeta_4", "n"] == 2
    assert pooled.loc["interaction", "n"] == 1
    assert pooled.loc["interaction", "mean"] == 4
    assert pd.isna(pooled.loc["interaction", "lo"])


def test_pending_factorial_reports_zero_pairs_without_dropping_original_arm():
    plan = _plan()
    rows = _rows(plan)
    rows = rows[(rows["zeta"] == 1) & (rows["score_noise"] == 2)]
    result = analysis.opinion_random_factorial(plan, rows)
    assert not result.empty and (result["n"] == 0).all()
    assert result["mean"].isna().all()
    plan.protocol = replace(plan.protocol, variant_arms=None)
    assert analysis.opinion_random_factorial(plan, rows).empty


def test_factorial_plot_labels_score_noise_and_retains_paired_counts(tmp_path, monkeypatch):
    plan = _plan()
    result = analysis.opinion_random_factorial(plan, _rows(plan))
    captured = []
    monkeypatch.setattr(plot, "plt_save_and_close", lambda fig, path: captured.append(fig))
    plot.plot_effects(result.rename(columns={"contrast": "arm"}), tmp_path, analysis.FACTORIAL_CONTRASTS)
    try:
        labels = [item.get_text() for item in captured[0].axes[0].get_yticklabels()]
        assert any("score σ" in item for item in labels)
        assert all(item.get_text() == "n=2" for item in captured[0].axes[0].texts)
    finally:
        for fig in captured:
            plt.close(fig)


def test_mixed_binary_provenance_keeps_reused_arms_and_rejects_conflicts():
    plan = _plan()
    plan.tasks = plan.tasks[:1]
    configurations = cfg.variant_arms(plan.protocol)
    original = {arm for family in plan.protocol.variants.values() for arm in family}
    binding = Binding(binary_sha256="old", arm_binary_sha256={
        name: "old" if name in original else "fixed" for name in configurations})
    arms = {name: fixture_arm(endpoints=(point(separation=.1),), raw_record=f"raw/{name}",
                      binary_sha256="fixed" if name not in original else None) for name in configurations}
    records = {plan.tasks[0].id: FullResult(arms=arms)}
    runs, trajectories = analysis.study_tables(plan, records, binding=binding)
    for table in (runs, trajectories, analysis.variant_arm_coverage(plan, records, binding=binding)):
        assert set(table.loc[table["arm"].isin(original), "binary_sha256"]) == {"old"}
        assert set(table.loc[~table["arm"].isin(original), "binary_sha256"]) == {"fixed"}
    arms[next(name for name in configurations if name not in original)].binary_sha256 = "wrong"
    with pytest.raises(ValueError, match="binary does not match"):
        analysis.study_tables(plan, records, binding=binding)
    with pytest.raises(ValueError, match="binary does not match"):
        analysis.variant_arm_coverage(plan, records, binding=binding)
    plan.protocol = replace(plan.protocol, variant_arms=None)
    records[plan.tasks[0].id].arms = {name: arms[name] for name in original}
    old_binding = Binding(binary_sha256="old")
    old_runs, _ = analysis.study_tables(plan, records, binding=old_binding)
    assert set(old_runs["binary_sha256"]) == {"old"}
