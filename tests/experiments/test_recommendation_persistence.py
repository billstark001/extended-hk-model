"""Small persistent runs verify paired histories and campaign lifecycle gates."""

import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from smp_bindings.model_state import load_snapshot
from smp_bindings.trajectory import TrajectoryReader

from common.experiments.plans import freeze_plan
from common.io.json import fingerprint, read_result, save_result
from ehk.micro.scenarios import create_scenario_metadata
from experiments.paper.recommendation import analysis, design, study


@pytest.fixture(scope="module")
def smp_binary(tmp_path_factory):
    binary = tmp_path_factory.mktemp("persistent-recommendation") / "smp"
    source = Path(__file__).resolve().parents[3] / "social-media-models"
    subprocess.run(["go", "build", "-o", str(binary), "./cmd/smp"], cwd=source, check=True)
    return binary


@pytest.fixture
def small_design(monkeypatch):
    for name, value in {
        "REFERENCE_HORIZON": 32,
        "FOLLOWUP": 16,
        "STAGE_CHECKPOINTS": (0, 16, 32),
        "FIXED_TRIGGER": 8,
        "LATEST_TRIGGER": 16,
    }.items():
        monkeypatch.setattr(design, name, value)
    metadata = create_scenario_metadata(
        "small-paired", influence=0.1, rewiring=0.1, repost_rate=0.1,
        recsys_type=design.REFERENCE, recsys_count=3, post_retain_count=2, max_sim_step=48,
    )
    metadata.update(
        NodeCount=10, NodeFollowCount=2, RecSysParams=dict(design.RECOMMENDER_PARAMS),
        RNG={"Algorithm": "pcg64-dxsm-v1", "Seed1": "0x0012", "Seed2": "0x0034"},
    )
    return {"id": "small-paired", "setting": "fixture", "sensitivity": "core", "replicate": 0, "metadata": metadata}


def _opinion_history(directory, arm):
    return np.asarray(TrajectoryReader(str(directory / arm["raw_record"])).channel("opinions")[:])


def _render_study(directory, profile, task, result):
    from experiments.paper.recommendation.plot import analyze

    freeze_plan(directory / "plan.json", {
        "schema": design.DESIGN_VERSION, "study": "paper_recommendation", "profile": profile,
        "protocol": design.design_protocol(profile), "tasks": [task],
    })
    save_result(directory / "records" / (task["id"] + ".json.gz"), {
        "task_fingerprint": fingerprint(task), "result": result,
    })
    output = directory / "analysis"
    analyze(directory, output)
    assert (output / "coverage.json").is_file()
    assert list(output.rglob("*.pdf")) and list(output.rglob("*.png"))


def test_staged_arms_replay_identical_prefix_and_keep_complete_records(
    smp_binary, tmp_path, small_design, monkeypatch
):
    monkeypatch.setattr(
        design, "select_stages", lambda points: dict(zip(("early", "middle", "late"), points))
    )
    result = study.run_task("staged", smp_binary, tmp_path, small_design)
    reference = _opinion_history(tmp_path, result["reference"])
    assert reference.shape == (33, 10)
    assert set(result["stages"]) == {"early", "middle", "late"}
    for phase, stage in result["stages"].items():
        start = stage["checkpoint"]["step"]
        expected = {"hold", "switch", "bridge", "hide_reposts"} if phase == "late" else {"hold", "switch"}
        assert set(stage["arms"]) == expected
        source = study._checkpoint_path(tmp_path, small_design, "reference", "reference", start)
        source_state = load_snapshot(str(source))
        for name, arm in stage["arms"].items():
            raw = tmp_path / arm["raw_record"]
            assert study._raw_complete(tmp_path, arm)
            history = _opinion_history(tmp_path, arm)
            assert history.shape == (start + design.FOLLOWUP + 1, 10)
            np.testing.assert_array_equal(history[: start + 1], reference[: start + 1])
            branch = study._checkpoint_path(tmp_path, small_design, phase, name, start)
            assert load_snapshot(str(branch)) == source_state
            assert arm["endpoints"][0]["step"] == start
            assert arm["endpoints"][0]["relative_step"] == 0
            assert arm["endpoints"][0]["exposure"]["updates"] == 0
            assert sum(point["exposure"]["updates"] for point in arm["endpoints"]) == 160
            metadata = json.loads((raw / "metadata.json").read_text())
            assert all(metadata[key] is True for key in ("AgentNumber", "OpinionSum", "RewiringEvent", "PostEvent"))
            assert metadata["TrajectoryPrecision"] == {"opinions": "float64", "opinion_sums": "float64"}
            if phase == "late":
                with gzip.open(tmp_path / arm["exposure_trace"]["path"], "rt") as stream:
                    rows = [json.loads(line) for line in stream]
                assert len(rows) == 160 and all(start < row["step"] <= start + 16 for row in rows)
    _render_study(tmp_path, "staged", small_design, result)


def test_variants_run_all_six_policies_from_one_initial_state(smp_binary, tmp_path, small_design):
    result = study.run_task("recommender_variants", smp_binary, tmp_path, small_design)
    arms = result["arms"]
    assert set(arms) == {"Opinion", "OpinionM9", "OpinionRandom", "Structure", "StructureM9", "StructureRandom"}
    initial = []
    for policy, arm in arms.items():
        assert study._raw_complete(tmp_path, arm)
        history = _opinion_history(tmp_path, arm)
        assert history.shape == (33, 10)
        initial.append(history[0])
        assert arm["endpoints"][0]["exposure"]["updates"] == 0
        assert sum(point["exposure"]["updates"] for point in arm["endpoints"]) == 320
        if policy != design.REFERENCE:
            assert arm["switch_step"] == 0
    assert all(np.array_equal(row, initial[0]) for row in initial)
    _render_study(tmp_path, "recommender_variants", small_design, result)


@pytest.mark.parametrize("passed", [False, True])
def test_campaign_releases_timing_only_after_stage_gate(tmp_path, monkeypatch, passed):
    binary = tmp_path / "binary"
    binary.write_bytes(b"frozen binary fixture")
    calls = []

    def run_one(profile, directory, selected_binary, workers, **kwargs):
        calls.append((profile, directory.name, workers))
        return {"missing": [], "state": "complete"}

    monkeypatch.setattr(study, "run_study", run_one)
    monkeypatch.setattr(analysis, "read_study", lambda directory, profile: ({"tasks": []}, {}))
    monkeypatch.setattr(analysis, "stage_gate", lambda tasks, records: {"passed": passed})
    result = study.run_campaign(tmp_path / "campaign", binary, 6)
    assert [row[0] for row in calls] == ["staged", "recommender_variants", *(["staged_timing"] if passed else [])]
    assert all(workers == 6 for _, _, workers in calls)
    assert result["staged_timing"]["state"] == ("complete" if passed else "deferred")
    gate = json.loads((tmp_path / "campaign" / "recommendation_runtime" / "timing_gate.json").read_text())
    assert gate["passed"] is passed


def test_study_validates_raw_cache_and_recovers_missing_summary(smp_binary, tmp_path, small_design, monkeypatch):
    monkeypatch.setattr(design, "generate_tasks", lambda profile: [small_design])
    first = study.run_study("recommender_variants", tmp_path, smp_binary, 1)
    assert first["completed"] == 1 and first["missing"] == []
    entry = read_result(tmp_path / "records" / (small_design["id"] + ".json.gz"))
    raw = tmp_path / entry["result"]["arms"]["Opinion"]["raw_record"]
    complete = raw / "observations.json"
    original = complete.read_bytes()
    complete.unlink()
    second = study.run_study("recommender_variants", tmp_path, smp_binary, 1)
    assert second["completed"] == 1 and complete.read_bytes() == original
    assert second["records"][0]["state"] == "complete"


def test_campaign_writes_interrupted_status_before_returning_control(tmp_path, monkeypatch):
    binary = tmp_path / "binary"
    binary.write_bytes(b"frozen binary fixture")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("interruption fixture")

    monkeypatch.setattr(study, "run_study", interrupted)
    directory = tmp_path / "campaign"
    with pytest.raises(KeyboardInterrupt, match="interruption fixture"):
        study.run_campaign(directory, binary, 6)
    status = json.loads((directory / "recommendation_runtime" / "status.json").read_text())
    assert status["state"] == "interrupted" and status["error"] == "interruption fixture"


def test_plan_only_lists_all_designs_without_starting_processes(small_design, monkeypatch, capsys):
    from experiments.paper.run import main

    monkeypatch.setattr(design, "generate_tasks", lambda profile: [small_design])

    def unexpected_run(*args, **kwargs):
        pytest.fail("plan-only started a simulation")

    monkeypatch.setattr(study, "run_study", unexpected_run)
    monkeypatch.setattr(study, "run_campaign", unexpected_run)
    main(["recommendation", "--plan-only", "--concurrency", "6"])
    plans = json.loads(capsys.readouterr().out)
    assert [plan["profile"] for plan in plans] == list(design.STUDIES)
    assert all(plan["protocol"]["recording"]["backend"] == "persistent_smp" for plan in plans)
