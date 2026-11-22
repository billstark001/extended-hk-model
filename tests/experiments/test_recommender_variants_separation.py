"""Canonical study boundaries preserve frozen plans, records and scientific outputs."""

import copy
import json
import subprocess

import pytest

from common.experiments.plans import freeze_plan
from common.io.json import atomic_json, fingerprint, save_result, sha
from experiments.paper.recommendation import analysis as staged_analysis
from experiments.paper.recommendation import study as staged_study
from experiments.paper.recommendation_shared import analysis as shared_analysis
from experiments.paper.recommendation_shared import runtime
from experiments.paper.recommender_variants import analysis, design, study

# Captured from the complete, unmodified plans before the package separation.
PLAN_FINGERPRINTS = {
    "staged": "83fbb8efcdd19767a01861a77469f024076f1f9c4d870156a69fa6b92b27cc78",
    "staged_timing": "1eda57d6d0bb51916b55417c906b570af048ba76b94fc68e789506a1ee820912",
    "recommender_variants": "62d3fc74502255e2ee65d4f94da546115d1290b8ffeb0096dfd906f2fe211767",
}
ORIGINAL_ARMS = (
    "Opinion", "OpinionM9", "OpinionRandom", "Structure", "StructureM9", "StructureRandom",
)


@pytest.fixture(autouse=True)
def forbid_native_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("compatibility checks must not launch a native process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.mark.parametrize("profile", PLAN_FINGERPRINTS)
def test_complete_plans_keep_original_fingerprints(profile):
    plan = study.make_plan() if profile == "recommender_variants" else staged_study.make_plan(profile)
    assert fingerprint(plan) == PLAN_FINGERPRINTS[profile]
    if profile == "recommender_variants":
        assert plan["tasks"] == design.generate_tasks()


@pytest.mark.parametrize("profile,namespace", [("staged", 0), ("staged_timing", 1), ("recommender_variants", 2)])
def test_seed_namespaces_and_sensitivity_initializations_stay_frozen(profile, namespace):
    plan = study.make_plan() if profile == "recommender_variants" else staged_study.make_plan(profile)
    tasks = plan["tasks"]
    core = [task for task in tasks if task["sensitivity"] == "core"]
    assert len(core) == 432
    assert len({fingerprint(task["metadata"]["RNG"]) for task in core}) == 432
    initial = namespace << 40
    assert core[0]["metadata"]["RNG"] == {
        "Algorithm": "pcg64-dxsm-v1",
        "Seed1": f"0x{0x45484B0000000000 + initial:016x}",
        "Seed2": f"0x{0xB9B7B4FFFFFFFFFF - initial:016x}",
    }
    for task in tasks:
        if task["sensitivity"] == "core":
            continue
        matched = next(other for other in core if other["replicate"] == task["replicate"]
                       and other["metadata"]["HKParams"]["Influence"] == .05
                       and other["metadata"]["HKParams"]["RewiringRate"] == .05
                       and other["metadata"]["HKParams"]["RepostRate"] == .1)
        assert task["metadata"]["RNG"] == matched["metadata"]["RNG"]


@pytest.mark.parametrize("factorial", [False, True])
def test_canonical_runner_honors_frozen_arm_registry_and_retained_sources(tmp_path, monkeypatch, factorial):
    plan = study.make_plan()
    protocol = copy.deepcopy(plan["protocol"])
    if not factorial:
        protocol.pop("variant_arms")
    arms = design.variant_arms(protocol)
    assert len(arms) == (9 if factorial else 6)
    assert set(ORIGINAL_ARMS) <= set(arms)
    task = plan["tasks"][0]
    binary = tmp_path / "binary"
    binary.write_bytes(b"binary fingerprint fixture; never executed")
    retained = {"arms": {"Opinion": {"raw_record": "raw/original/full/Opinion", "binary_sha256": "old"}}}
    provenance = {name: "old" if name in ORIGINAL_ARMS else sha(binary) for name in arms}
    calls = []

    def paired(selected_binary, directory, selected_task, phase, start, horizon, selected_arms, **kwargs):
        calls.append((selected_binary, directory, selected_task, phase, start, horizon, selected_arms, kwargs))
        return retained["arms"]

    monkeypatch.setattr(runtime, "paired", paired)
    expected = {"arms": retained["arms"]}
    assert study.run_task(binary, tmp_path, task, protocol=protocol, existing=retained,
                          arm_binary_sha256=provenance) == expected
    assert len(calls) == 1
    selected_binary, directory, selected_task, phase, start, horizon, selected_arms, kwargs = calls[0]
    assert (selected_binary, directory, selected_task, phase, start, horizon) == (
        binary, tmp_path, task, "full", 0, 2048,
    )
    assert set(selected_arms) == set(arms)
    assert selected_arms["OpinionM9"] is None
    assert selected_arms["StructureM9"] == {"recommender": "StructureM9", "at": 0}
    assert kwargs["parameters"] == {name: spec["parameters"] for name, spec in arms.items()}
    assert kwargs["existing"] is retained["arms"]
    assert kwargs["arm_binary_sha256"] is provenance


def _record_fixture(factorial):
    plan = study.make_plan()
    plan["tasks"] = [plan["tasks"][0], plan["tasks"][31]]
    if not factorial:
        plan["protocol"].pop("variant_arms")
    arms = design.variant_arms(plan["protocol"])
    binding = {"arm_binary_sha256": {
        name: "original-six-binary" if name in ORIGINAL_ARMS else "factorial-binary" for name in arms
    }}
    records = {}
    for repeat, task in enumerate(plan["tasks"]):
        recorded = {}
        for index, name in enumerate(arms):
            points = []
            for step in (0, 16, 32):
                exposure = dict.fromkeys(shared_analysis.EXPOSURE_COUNTS, 0)
                exposure.update({source: dict.fromkeys(shared_analysis.SLOT_MEASURES, 0)
                                 for source in shared_analysis.SLOT_SOURCES})
                if step:
                    exposure.update(updates=8000, requested=80000, recommended=79000,
                                    cross_camp=800 + 10 * index, rewires=20 + repeat,
                                    absolute_shift=1 + index / 10, neighbor_with_posts=8000,
                                    neighbor_share_sum=4000)
                    exposure["policy"].update(shown=71000, cross_camp=700 + 10 * index)
                    exposure["exploration"].update(shown=8000, cross_camp=100)
                separation = .1 if step == 0 else (.45 if index % 2 == 0 else .2)
                excess = .05 if step == 0 else (.08 if index % 2 == 0 else .12)
                if step == 32:
                    separation, excess = .4 + .03 * index + .01 * repeat, .14 + .005 * index
                points.append({
                    "step": step, "relative_step": step, "exposure": exposure,
                    "pathway": {"separation": separation, "excess_concordance": excess,
                                "edge_concordance": .5 + excess, "pair_concordance": .5},
                })
            recorded[name] = {
                "endpoints": points, "terminal": {"status": "absorbed", "k_major": 2},
                "final_opinions": [-.8, -.7, .7, .8],
                "raw_record": f"raw/{task['id']}/full/{name}",
                **({"binary_sha256": "factorial-binary"} if name not in ORIGINAL_ARMS else {}),
            }
        records[task["id"]] = {"arms": recorded}
    return plan, records, binding


@pytest.mark.parametrize("factorial", [False, True])
def test_frozen_six_and_nine_arm_records_keep_readings_and_binary_provenance(tmp_path, factorial):
    plan, records, binding = _record_fixture(factorial)
    frozen = freeze_plan(tmp_path / "plan.json", plan)
    atomic_json(tmp_path / "runtime/binding.json", binding)
    for task in plan["tasks"]:
        save_result(tmp_path / "records" / (task["id"] + ".json.gz"), {
            "task_fingerprint": fingerprint(task), "result": records[task["id"]],
        })
    original_files = {path: sha(path) for path in tmp_path.rglob("*") if path.is_file()}
    new_plan, new_records = analysis.read_study(tmp_path)
    assert new_plan == frozen
    assert new_records == records
    runs, trajectories = analysis.study_tables(new_plan, new_records, binding=binding)
    coverage = analysis.variant_arm_coverage(plan, records, binding=binding)
    contrasts = analysis.contrast_table(runs)
    factorial_table = analysis.opinion_random_factorial(plan, runs)
    assert coverage["completed"].sum() == len(runs)
    assert (coverage["completed"] == coverage["planned"]).all()
    assert len(contrasts) == 84
    opinion_pair = contrasts[(contrasts["setting"] == "all") & (contrasts["arm"] == "Opinion")
                             & (contrasts["control"] == "OpinionM9")
                             & (contrasts["metric"] == "separation_change")]
    assert opinion_pair.iloc[0]["n"] == 2
    assert opinion_pair.iloc[0]["mean"] == pytest.approx(-.03)
    assert len(runs) == (18 if factorial else 12)
    assert len(trajectories) == len(runs) * 3
    assert set(runs.loc[runs["arm"].isin(ORIGINAL_ARMS), "binary_sha256"]) == {"original-six-binary"}
    assert factorial_table.empty is (not factorial)
    if factorial:
        assert set(runs.loc[~runs["arm"].isin(ORIGINAL_ARMS), "binary_sha256"]) == {"factorial-binary"}
        pooled = factorial_table[(factorial_table["setting"] == "all")
                                 & (factorial_table["metric"] == "separation_change")].set_index("contrast")
        assert pooled.loc["interaction", "n"] == 2
        assert pooled.loc["interaction", "mean"] == pytest.approx(.15)
    assert all(sha(path) == digest for path, digest in original_files.items())


@pytest.mark.parametrize("passed", [False, True])
def test_staged_campaign_releases_timing_without_waiting_for_variants(tmp_path, monkeypatch, passed):
    binary = tmp_path / "binary"
    binary.write_bytes(b"fingerprint fixture")
    calls = []

    def run_staged(profile, directory, selected_binary, workers, **kwargs):
        assert profile != "recommender_variants"
        calls.append(profile)
        return {"missing": [], "state": "complete"}

    def unavailable_variants(*args, **kwargs):
        pytest.fail("staged campaign depends on the independent variants runner")

    monkeypatch.setattr(staged_study, "_preflight", lambda *args: None)
    monkeypatch.setattr(staged_study, "run_study", run_staged)
    monkeypatch.setattr(study, "run_study", unavailable_variants)
    monkeypatch.setattr(staged_analysis, "read_study", lambda *args: ({"tasks": []}, {}))
    monkeypatch.setattr(staged_analysis, "stage_gate", lambda *args: {"passed": passed})
    outcomes = staged_study.run_campaign(tmp_path / "campaign", binary, 6)
    assert calls == ["staged", *(["staged_timing"] if passed else [])]
    assert set(outcomes) == {"staged", "staged_timing"}
    assert outcomes["staged_timing"]["state"] == ("complete" if passed else "deferred")


def test_independent_variants_cli_prints_its_frozen_plan(monkeypatch, capsys):
    from experiments.paper.recommender_variants.study import main

    def forbidden(*args, **kwargs):
        pytest.fail("plan-only invoked a simulation or the staged campaign")

    monkeypatch.setattr(study, "run_study", forbidden)
    monkeypatch.setattr(staged_study, "run_campaign", forbidden)
    main(["--plan-only"])
    assert json.loads(capsys.readouterr().out) == study.make_plan()


def test_independent_variants_cli_runs_without_stage_campaign(tmp_path, monkeypatch):
    from experiments.paper.recommender_variants.study import main

    calls = []
    directory, binary = tmp_path / "variants", tmp_path / "binary"
    expected = {"state": "complete", "missing": []}

    def run_variants(*args):
        calls.append(args)
        return expected

    def forbidden(*args, **kwargs):
        pytest.fail("the independent implementation comparison invoked the staged campaign")

    monkeypatch.setattr(study, "run_study", run_variants)
    monkeypatch.setattr(staged_study, "run_campaign", forbidden)
    assert main([str(directory), "--smp-binary", str(binary), "--concurrency", "3"]) == expected
    assert calls == [(directory, binary, 3)]


@pytest.mark.parametrize("profile", ("staged", "staged_timing", "recommender_variants"))
def test_canonical_readers_reject_other_study_profiles_even_when_requested_explicitly(tmp_path, profile):
    plan = study.make_plan() if profile == "recommender_variants" else staged_study.make_plan(profile)
    plan["tasks"] = plan["tasks"][:1]
    freeze_plan(tmp_path / "plan.json", plan)
    owner = analysis if profile == "recommender_variants" else staged_analysis
    other = staged_analysis if profile == "recommender_variants" else analysis
    assert owner.read_study(tmp_path)[0]["profile"] == profile
    with pytest.raises(ValueError, match="unexpected experiment"):
        other.read_study(tmp_path)
    with pytest.raises(ValueError, match="unexpected experiment"):
        other.read_study(tmp_path, profile)
    with pytest.raises(ValueError, match="profile"):
        other.study_tables(plan, {})


def test_stage_entry_rejects_implementation_comparison_before_scheduling(tmp_path):
    with pytest.raises(ValueError):
        staged_study.make_plan("recommender_variants")
    with pytest.raises(ValueError):
        staged_study.run_task("recommender_variants", tmp_path / "binary", tmp_path, {})
    with pytest.raises(ValueError):
        staged_study.run_study("recommender_variants", tmp_path, tmp_path / "binary", 1)
