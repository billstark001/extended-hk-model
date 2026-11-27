"""Small persistent runs verify paired histories and campaign lifecycle gates."""

import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from common.experiments.lifecycle import RunStatus
from smp_bindings.model_state import load_snapshot
from smp_bindings.trajectory import TrajectoryReader

from ehk.domain import (freeze_plan,read_plan,Task,FullResult,StagedResult,Plan,Rng,StageGate,to_wire)
from dataclasses import replace
from tests.experiments import recommendation_fixtures as fixtures
from common.io.json import sha
from ehk.micro.scenarios import create_scenario_metadata
from experiments.paper.recommendation import analysis, design, study
from experiments.paper.recommendation_shared import runtime
from experiments.paper.recommendation_shared.storage import StudyStore, TASKS
from common.storage.documents import DocumentStore
from experiments.paper.recommender_variants import design as variants_design
from experiments.paper.recommender_variants import study as variants_study


@pytest.fixture(scope="module")
def smp_binary(tmp_path_factory):
    binary = tmp_path_factory.mktemp("persistent-recommendation") / "smp.exe"
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
    monkeypatch.setattr(variants_design, "REFERENCE_HORIZON", 32)
    metadata = create_scenario_metadata(
        "small-paired", influence=0.1, rewiring=0.1, repost_rate=0.1,
        recsys_type=design.REFERENCE, recsys_count=3, post_retain_count=2, max_sim_step=48,
    )
    metadata=replace(metadata,NodeCount=10,NodeFollowCount=2,RecSysParams=replace(design.RECOMMENDER_PARAMS),
        RNG=Rng(Algorithm="pcg64-dxsm-v1",Seed1="0x0012",Seed2="0x0034"))
    return Task(id="small-paired",setting="fixture",sensitivity="core",replicate=0,metadata=metadata)


def _opinion_history(directory, arm):
    return np.asarray(TrajectoryReader(str(directory / arm.raw_record)).channel("opinions")[:])


def _render_study(directory, profile, task, result):
    if profile == "recommender_variants":
        from experiments.paper.recommender_variants.plot import analyze
        protocol = variants_design.design_protocol()
    else:
        from experiments.paper.recommendation.plot import analyze
        protocol = design.design_protocol(profile)

    plan=freeze_plan(directory/"plan.json",Plan(schema=design.DESIGN_VERSION,study="paper_recommendation",
        profile=profile,protocol=protocol,tasks=(task,)))
    with StudyStore(directory) as store:
        store.put_task(task, result, plan=plan)
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
    reference = _opinion_history(tmp_path, result.reference)
    assert reference.shape == (33, 10)
    assert set(result.stages) == {"early", "middle", "late"}
    for phase, stage in result.stages.items():
        start = stage.checkpoint.step
        expected = {"hold", "switch", "bridge", "hide_reposts"} if phase == "late" else {"hold", "switch"}
        assert set(stage.arms) == expected
        source = runtime.checkpoint_path(tmp_path, small_design, "reference", "reference", start)
        source_state = load_snapshot(str(source))
        for name, arm in stage.arms.items():
            raw = tmp_path / arm.raw_record
            assert runtime.complete_raw(tmp_path, arm)
            history = _opinion_history(tmp_path, arm)
            assert history.shape == (start + design.FOLLOWUP + 1, 10)
            np.testing.assert_array_equal(history[: start + 1], reference[: start + 1])
            branch = runtime.checkpoint_path(tmp_path, small_design, phase, name, start)
            assert load_snapshot(str(branch)) == source_state
            assert arm.endpoints[0].step == start
            assert arm.endpoints[0].relative_step == 0
            assert arm.endpoints[0].exposure.updates == 0
            assert sum(point.exposure.updates for point in arm.endpoints) == 160
            metadata = json.loads((raw / "metadata.json").read_text())
            assert all(metadata[key] is True for key in ("AgentNumber", "OpinionSum", "RewiringEvent", "PostEvent"))
            assert metadata["TrajectoryPrecision"] == {"opinions": "float64", "opinion_sums": "float64"}
            if phase == "late":
                with gzip.open(tmp_path / arm.exposure_trace.path, "rt") as stream:
                    rows = [json.loads(line) for line in stream]
                assert len(rows) == 160 and all(start < row["step"] <= start + 16 for row in rows)
    _render_study(tmp_path, "staged", small_design, result)


def test_variants_and_factorial_arms_share_initial_state_and_rng(smp_binary, tmp_path, small_design):
    result = variants_study.run_task(smp_binary, tmp_path, small_design)
    arms = result.arms
    assert set(arms) == set(variants_design.variant_arms()) and len(arms) == 9
    initial = []
    for policy, arm in arms.items():
        assert runtime.complete_raw(tmp_path, arm)
        history = _opinion_history(tmp_path, arm)
        assert history.shape == (33, 10)
        initial.append(history[0])
        assert arm.endpoints[0].exposure.updates == 0
        assert sum(point.exposure.updates for point in arm.endpoints) == 320
        if policy != design.REFERENCE:
            assert arm.switch_step == 0
        parameters = json.loads((tmp_path / arm.raw_record / "metadata.json").read_text())["RecSysParams"]
        assert parameters == to_wire(small_design.metadata.RecSysParams.with_overrides(variants_design.variant_arms()[policy].parameters))
        checkpoint = runtime.checkpoint_path(tmp_path, small_design, "full", policy, 0)
        baseline = runtime.checkpoint_path(tmp_path, small_design, "full", "OpinionRandom", 0)
        assert load_snapshot(str(checkpoint)) == load_snapshot(str(baseline))
    assert all(np.array_equal(row, initial[0]) for row in initial)
    _render_study(tmp_path, "recommender_variants", small_design, result)


def test_retained_arm_cannot_be_recomputed_with_a_replacement_binary(tmp_path, small_design, monkeypatch):
    binary = tmp_path / "smp"
    binary.write_bytes(b"replacement binary fixture")
    monkeypatch.setattr(runtime, "complete_raw", lambda *args: False)

    def unexpected_native(*args, **kwargs):
        pytest.fail("retained arm was recomputed with a different native binary")

    monkeypatch.setattr(runtime, "observe", unexpected_native)
    with pytest.raises(ValueError, match="original binary"):
        variants_study.run_task(
            binary, tmp_path, small_design,
            existing=FullResult(arms={"Opinion":fixtures.arm(raw_record="raw/missing")}),
            arm_binary_sha256=dict.fromkeys(variants_design.variant_arms(), "original-binary"),
        )


@pytest.mark.parametrize("passed", [False, True])
def test_campaign_releases_timing_only_after_stage_gate(tmp_path, monkeypatch, passed):
    binary = tmp_path / "binary"
    binary.write_bytes(b"frozen binary fixture")
    calls = []
    monkeypatch.setattr(study, "_preflight", lambda binary, control: None)

    def run_one(profile, directory, selected_binary, workers, **kwargs):
        calls.append((profile, directory.name, workers))
        return RunStatus(planned=0,completed=0,missing=(),state="complete",records=())

    monkeypatch.setattr(study, "run_study", run_one)
    monkeypatch.setattr(analysis, "read_study", lambda directory,profile:(replace(study.make_plan("staged"),tasks=()),{}))
    monkeypatch.setattr(analysis, "stage_gate", lambda tasks,records:StageGate(n=2,mean=.1,lo=.05,hi=.15,planned_core=2,completed_core=2,complete=True,passed=passed,minimum_effect=.05))
    result = study.run_campaign(tmp_path / "campaign", binary, 6)
    assert [row[0] for row in calls] == ["staged", *(["staged_timing"] if passed else [])]
    assert all(workers == 6 for _, _, workers in calls)
    assert result["staged_timing"].state == ("complete" if passed else "deferred")
    gate = json.loads((tmp_path / "campaign" / "recommendation_runtime" / "timing_gate.json").read_text())
    assert gate["passed"] is passed


def test_study_validates_raw_cache_and_recovers_missing_summary(smp_binary, tmp_path, small_design, monkeypatch):
    monkeypatch.setattr(variants_design, "generate_tasks", lambda profile=None: [small_design])
    first = variants_study.run_study(tmp_path, smp_binary, 1)
    assert first.completed == 1 and first.missing == ()
    plan = read_plan(tmp_path / "plan.json")
    with StudyStore(tmp_path, readonly=True) as store:
        entry = store.get_task(small_design, plan)
    raw = tmp_path / entry.result.arms["Opinion"].raw_record
    original = {path: sha(path) for path in raw.iterdir() if path.is_file() and path.name not in ("smp.log", "job.lock") and not path.name.endswith(("-shm", "-wal"))}
    with DocumentStore(tmp_path / "analysis.sqlite") as store:
        with store.connection:
            store.connection.execute("DELETE FROM documents WHERE namespace=?", (TASKS,))
    second = variants_study.run_study(tmp_path, smp_binary, 1)
    assert second.completed == 1 and all(sha(path) == digest for path, digest in original.items())
    assert not (raw / "observations.json").exists()
    assert second.records[0].state == "complete"
    entry.result.arms["Opinion"].binary_sha256 = "conflicting-binary-fixture"
    with StudyStore(tmp_path) as store:
        store.put_task(small_design, entry.result, plan=plan)
    rejected = variants_study.run_study(tmp_path, smp_binary, 1)
    assert rejected.completed == 0 and rejected.missing == (small_design.id,)
    assert rejected.records[0].state == "failed"
    assert "binary provenance" in rejected.records[0].error


def test_campaign_writes_interrupted_status_before_returning_control(tmp_path, monkeypatch):
    binary = tmp_path / "binary"
    binary.write_bytes(b"frozen binary fixture")
    monkeypatch.setattr(study, "_preflight", lambda binary, control: None)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("interruption fixture")

    monkeypatch.setattr(study, "run_study", interrupted)
    directory = tmp_path / "campaign"
    with pytest.raises(KeyboardInterrupt, match="interruption fixture"):
        study.run_campaign(directory, binary, 6)
    status = json.loads((directory / "recommendation_runtime" / "status.json").read_text())
    assert status["state"] == "interrupted" and status["error"] == "interruption fixture"


def test_campaign_stops_before_scheduling_when_volume_preflight_fails(tmp_path, monkeypatch):
    binary = tmp_path / "binary"
    binary.write_bytes(b"frozen binary fixture")

    def failed_preflight(binary, control):
        raise RuntimeError("volume persistence failed")

    def unexpected_run(*args, **kwargs):
        pytest.fail("campaign scheduled experiments after a failed persistence check")

    monkeypatch.setattr(study, "_preflight", failed_preflight)
    monkeypatch.setattr(study, "run_study", unexpected_run)
    directory = tmp_path / "campaign"
    with pytest.raises(RuntimeError, match="volume persistence failed"):
        study.run_campaign(directory, binary, 6)
    status = json.loads((directory / "recommendation_runtime" / "status.json").read_text())
    assert status["state"] == "failed" and status["results"] == {}


def test_campaign_preflight_writes_and_reopens_a_small_persistent_run(smp_binary, tmp_path):
    study._preflight(smp_binary, tmp_path)
    raw = tmp_path / "preflight" / "storage-check"
    metadata = json.loads((raw / "metadata.json").read_text())
    assert metadata["NodeCount"] == 8 and metadata["MaxSimulationStep"] == 2
    assert not (raw / "observations.json").exists()
    assert np.asarray(TrajectoryReader(str(raw)).channel("opinions")[:]).shape == (3, 8)


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


def test_explicit_job_migration_preserves_full_checkpoint(smp_binary, tmp_path, small_design):
    from experiments.paper.recommendation_shared.migrate_jobs import migrate_jobs
    from common.io.json import atomic_json
    from common.runtime.failures import ProtocolError

    first = runtime.observe(smp_binary, tmp_path, small_design, "full", "hold", [0, 16], checkpoints=[0, 16])
    raw = tmp_path / first.raw_record
    jobs = [raw / "job.json", raw.parent / ".jobs/hold.json"]
    checkpoints = {path: sha(path) for path in raw.glob("checkpoint-*.msgpack.lz4")}
    for path in jobs:
        old = json.loads(path.read_text())
        old.pop("result_protocol")
        old["summary_path"] = str(raw / "observations.json")
        atomic_json(path, old)
    with pytest.raises(ProtocolError):
        runtime.observe(smp_binary, tmp_path, small_design, "full", "hold", [0, 16], checkpoints=[0, 16])
    receipt = migrate_jobs(tmp_path)
    assert len(receipt.jobs) == 2 and not receipt.scientific_state_changed
    assert runtime.observe(smp_binary, tmp_path, small_design, "full", "hold", [0, 16], checkpoints=[0, 16]) == first
    assert all(sha(path) == digest for path, digest in checkpoints.items())
    assert not (raw / "observations.json").exists()


def test_staged_partial_arms_commit_for_resume_without_entering_analysis(
    smp_binary, tmp_path, small_design, monkeypatch,
):
    monkeypatch.setattr(design, "generate_tasks", lambda profile: [small_design])
    monkeypatch.setattr(design, "select_stages", lambda points: dict(zip(design.STAGES, points)))
    original = runtime.observe
    calls = []

    def interrupted(binary, directory, task, phase, arm, *args, **kwargs):
        calls.append((phase, arm))
        if (phase, arm) == ("early", "switch"):
            raise KeyboardInterrupt("staged partial fixture")
        return original(binary, directory, task, phase, arm, *args, **kwargs)

    monkeypatch.setattr(runtime, "observe", interrupted)
    with pytest.raises(KeyboardInterrupt, match="staged partial fixture"):
        study.run_study("staged", tmp_path, smp_binary, 1)
    plan, complete = analysis.read_study(tmp_path, "staged")
    assert complete == {}
    with StudyStore(tmp_path, readonly=True) as store:
        assert store.get_task(small_design, plan) is None
        partial = store.get_task(small_design,plan,pending=True).result
    assert set(partial.stages["early"].arms) == {"hold"}
    assert partial.reference.stopped_at == 32
    old_files = {path: sha(path) for path in (tmp_path / "raw").rglob("*")
                 if path.is_file() and path.name not in ("smp.log", "job.lock") and not path.name.endswith(("-wal", "-shm"))}
    calls.clear()

    def resumed(binary, directory, task, phase, arm, *args, **kwargs):
        calls.append((phase, arm))
        return original(binary, directory, task, phase, arm, *args, **kwargs)

    monkeypatch.setattr(runtime, "observe", resumed)
    assert study.run_study("staged",tmp_path,smp_binary,1).completed == 1
    assert ("reference", "reference") not in calls and ("early", "hold") not in calls
    assert calls[0] == ("early", "switch")
    assert all(sha(path) == digest for path, digest in old_files.items())
    with StudyStore(tmp_path, readonly=True) as store:
        assert store.get_task(small_design, plan, pending=True) is None
        assert store.get_task(small_design, plan) is not None
    assert len(analysis.read_study(tmp_path, "staged")[1]) == 1


def test_timing_pending_arms_resume_without_false_completion(smp_binary, tmp_path, small_design, monkeypatch):
    from experiments.paper.recommendation_shared.analysis import read_study as read_observations

    monkeypatch.setattr(design, "generate_tasks", lambda profile: [small_design])
    monkeypatch.setattr(analysis, "read_study", lambda *args:(replace(study.make_plan("staged"),tasks=(),plan_fingerprint="gate-fixture"),{}))
    monkeypatch.setattr(analysis, "stage_gate", lambda *args:StageGate(n=2,mean=.1,lo=.05,hi=.15,planned_core=2,completed_core=2,complete=True,passed=True,minimum_effect=.05))
    original = runtime.observe

    def interrupted(binary, directory, task, phase, arm, *args, **kwargs):
        if arm == "always_structure":
            raise KeyboardInterrupt("timing partial fixture")
        return original(binary, directory, task, phase, arm, *args, **kwargs)

    monkeypatch.setattr(runtime, "observe", interrupted)
    with pytest.raises(KeyboardInterrupt, match="timing partial fixture"):
        study.run_study("staged_timing", tmp_path, smp_binary, 1)
    plan, records = read_observations(tmp_path, "staged_timing")
    assert records == {}
    with StudyStore(tmp_path, readonly=True) as store:
        assert set(store.get_task(small_design,plan,pending=True).result.arms) == {"hold"}
    calls = []

    def resumed(binary, directory, task, phase, arm, *args, **kwargs):
        calls.append(arm)
        return original(binary, directory, task, phase, arm, *args, **kwargs)

    monkeypatch.setattr(runtime, "observe", resumed)
    assert study.run_study("staged_timing",tmp_path,smp_binary,1).completed == 1
    assert calls == ["always_structure", "fixed", "state"]
    assert set(read_observations(tmp_path, "staged_timing")[1][small_design.id].arms) == {"hold", "always_structure", "fixed", "state"}
