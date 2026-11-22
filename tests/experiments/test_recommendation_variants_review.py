"""Factorial figures preserve matched cohorts and compose without source data."""

import json
import shutil

import pandas as pd
import pytest

from common.experiments.plans import freeze_plan
from common.io.json import atomic_json, sha
from experiments.paper.recommendation_shared.analysis import EFFECTS
from experiments.paper.recommender_variants import design, variants_review


def bundle_fixture(tmp_path):
    bundle = tmp_path / "bundle"
    tasks = design.generate_tasks()
    tasks = [tasks[0], tasks[6], tasks[3]]
    plan = freeze_plan(bundle / "data/study/plan.json", {
        "schema": design.DESIGN_VERSION, "study": "paper_recommendation", "profile": "recommender_variants",
        "protocol": design.design_protocol(), "tasks": tasks,
    })
    rows = []
    for index, task in enumerate(tasks):
        for arm, spec in design.variant_arms(plan["protocol"]).items():
            if index == 2 and arm == "OpinionRandomZeta4Noise0":
                continue
            metadata = task["metadata"]
            params = {**metadata["RecSysParams"], **spec["parameters"]}
            value = .1 + .04 * params["Steepness"] + .02 * params["OpRandomNoiseStd"]
            rows.append({
                "task_id": task["id"], "arm": arm, "stage": "full", "recommender": spec["recommender"],
                "zeta": params["Steepness"], "score_noise": params["OpRandomNoiseStd"],
                "sensitivity": "core", "epsilon": .45, "history": 0, "setting": task["setting"],
                "influence": metadata["HKParams"]["Influence"], "rewiring": metadata["HKParams"]["RewiringRate"],
                "repost": metadata["HKParams"]["RepostRate"], "start_separation": 0., "start_excess": 0.,
                "start_step": 0, "replicate": task["replicate"], "bridge_shown": 0,
                "cross_confident": 0, "updates": 100, "outcome": "consensus" if index else "two groups",
                "end_separation": value, "end_excess": value / 2, "pathway": "PbS" if index else "SbP",
                **dict.fromkeys(EFFECTS, value),
            })
    table = bundle / "data/tables/runs.csv.gz"
    table.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(table, index=False)
    atomic_json(bundle / "data/manifest.json", {
        "schema": 1, "completed": 2,
        "files": {str(path.relative_to(bundle / "data")): sha(path)
                  for path in (table, bundle / "data/study/plan.json")},
    })
    return bundle, pd.DataFrame(rows)


def test_matched_cohort_drops_partial_configuration_and_rejects_initial_mismatch(tmp_path):
    _, rows = bundle_fixture(tmp_path)
    matched = variants_review.matched_factorial_runs(rows)
    assert matched["task_id"].nunique() == 2
    assert matched.groupby("arm")["task_id"].nunique().eq(2).all()
    rows.loc[0, "start_separation"] = .5
    with pytest.raises(ValueError, match="different initial"):
        variants_review.matched_factorial_runs(rows)
    with pytest.raises(ValueError, match="duplicate"):
        variants_review.matched_factorial_runs(pd.concat([matched, matched.iloc[:1]]))


def test_prepare_preserves_layout_and_compose_uses_only_plot_data(tmp_path):
    bundle, _ = bundle_fixture(tmp_path)
    output = tmp_path / "plots"
    assert variants_review.prepare(bundle, output) == {"matched": 2, "planned": 3}
    data = json.loads((output / "plot_data.json").read_text())
    assert len(data["strata"]) == 1
    assert len(data["arms"]) == 5
    matched = pd.read_csv(output / "matched_runs.csv.gz")
    assert matched["task_id"].nunique() == 2
    assert matched.groupby("arm")["task_id"].nunique().eq(2).all()
    means = pd.read_csv(output / "matched_arm_means.csv")
    assert set(means.loc[means["setting"] == "all", "n"]) == {2}
    pairs = pd.read_csv(output / "paired_vs_m9.csv.gz")
    assert len(pairs) == 2 * 8
    assert pairs.groupby("arm")["task_id"].nunique().eq(2).all()
    counts = pd.read_csv(output / "matched_pathway_counts.csv")
    assert counts.groupby("arm")["n"].sum().eq(2).all()
    effects = pd.DataFrame(data["strata"][0]["factorial"])
    assert set(effects.loc[effects["setting"] == "all", "n"]) == {2}
    layout = json.loads((output / "layout.json").read_text())
    layout["summary_size"] = [8, 6]
    layout["strata"] = ["core_e45_kh0"]
    atomic_json(output / "layout.json", layout)
    variants_review.prepare(bundle, output)
    assert json.loads((output / "layout.json").read_text()) == layout
    data["strata"].append({**data["strata"][0], "name": "excluded"})
    atomic_json(output / "plot_data.json", data)
    shutil.rmtree(bundle)
    variants_review.compose(output)
    assert (output / "figures/core_e45_kh0_summary.pdf").is_file()
    assert (output / "figures/core_e45_kh0_cells.png").is_file()
    assert "<text" in (output / "figures/core_e45_kh0_summary.svg").read_text()
    assert not (output / "figures/excluded_summary.pdf").exists()
    assert json.loads((output / "figure_manifest.json").read_text())["plot_data_sha256"] == sha(output / "plot_data.json")
    layout["strata"] = ["unknown"]
    atomic_json(output / "layout.json", layout)
    with pytest.raises(ValueError, match="available strata"):
        variants_review.compose(output)


def test_prepare_rejects_changed_authenticated_tables(tmp_path):
    bundle, _ = bundle_fixture(tmp_path)
    with (bundle / "data/tables/runs.csv.gz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="portable data changed"):
        variants_review.prepare(bundle, tmp_path / "plots")
