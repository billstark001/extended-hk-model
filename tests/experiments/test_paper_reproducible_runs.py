"""New-run identity and process-ablation contracts protect historical inputs."""

import json
from dataclasses import replace
from common.io.json import atomic_json, canonical
from ehk.domain import Coordinates, to_wire

import pytest

from ehk.micro.simulation import SimulationAdapter
from experiments.paper.pathway_controls.design import ARMS, arm_metadata
from experiments.paper.pathway_controls.design import make_plan as control_plan
from experiments.paper.scenarios import execution_tasks, generate_tasks
from experiments.paper.study import make_plan


def test_seeded_execution_retains_historical_names_and_has_independent_coordinates():
    old, new = generate_tasks("gradation"), execution_tasks("gradation")
    assert [task.id for task in new] == [metadata.UniqueName for metadata in old]
    seeds = [canonical(task.metadata.RNG) for task in new]
    assert len(seeds) == len(set(seeds))
    assert new[0].coordinates == Coordinates(replicate=0, rewiring=.005, influence=.005,
                                              repost=0., recommendation="st")
    assert all(metadata.RNG is None for metadata in old)
    assert execution_tasks("mech") == execution_tasks("mech")
    assert execution_tasks("mech", seed_namespace="independent")[0].metadata.RNG != execution_tasks("mech")[0].metadata.RNG
    plan = make_plan("mech")
    assert json.loads(canonical(plan))["tasks"][0]["metadata"]["RNG"]


def test_control_arms_share_initial_rng_and_change_only_the_intended_process():
    plan = control_plan(replicates=2, cases=["baseline"], horizon=16, node_count=16, follow_count=4)
    first, second = plan.tasks
    arms = {arm: arm_metadata(first, arm) for arm in ARMS}
    assert arms["full"].HKParams.Influence > 0
    assert arms["full"].HKParams.RewiringRate > 0
    assert arms["q_zero"].HKParams == replace(arms["full"].HKParams, RewiringRate=0.)
    assert arms["alpha_zero"].HKParams == replace(arms["full"].HKParams, Influence=0.)
    assert all(value.RNG == first.metadata.RNG for value in arms.values())
    assert first.metadata.RNG != second.metadata.RNG
    for arm, metadata in arms.items():
        assert replace(metadata, UniqueName=first.metadata.UniqueName, HKParams=first.metadata.HKParams) == first.metadata
    assert plan.protocol.snapshot_steps == (0,16)
    assert json.loads(canonical(plan))["protocol"]["snapshot_steps"] == [0,16]


def test_seeded_runner_cannot_cache_unseeded_or_differently_seeded_output(tmp_path):
    task = execution_tasks("mech")[0]
    directory = tmp_path / task.id
    directory.mkdir()
    (directory / "finished-000001.msgpack").touch()
    path = directory / "metadata.json"
    atomic_json(path, generate_tasks("mech")[0])
    adapter = SimulationAdapter(tmp_path / "smp", tmp_path)
    with pytest.raises(ValueError, match="existing simulation differs.*RNG"):
        adapter.is_cached(task)
    # The actual native wire may omit an optional false event field.
    actual = json.loads(canonical(task.metadata))
    actual.pop("ViewPostsDetails")
    atomic_json(path, actual)
    assert adapter.is_cached(task)
    actual["RNG"]["Seed1"] = "0x0000000000000000"
    atomic_json(path, actual)
    with pytest.raises(ValueError, match="existing simulation differs in RNG"):
        adapter.is_cached(task)


def test_original_run_plan_prints_without_resolving_private_workspace(capsys):
    from experiments.paper.run import main

    main(["mech", "--plan-only"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["schema"] == 2 and plan["tasks"][0]["metadata"]["RNG"]


def test_primary_smp_adapter_isolates_mutable_metadata_in_parallel(tmp_path, monkeypatch):
    from copy import deepcopy
    from common.experiments.execution import execute_tasks
    import smp_bindings.simulation as binding

    tasks = execution_tasks("mech")[:2]
    originals = deepcopy([task.metadata for task in tasks])
    finished = set()

    def single(binary, workspace, metadata, show_progress, print_lock, **kwargs):
        metadata.RNG.Seed1 = "0xffffffffffffffff"
        finished.add(metadata.UniqueName)
        return metadata.UniqueName

    monkeypatch.setattr(binding, "run_simulation", single)
    monkeypatch.setattr(binding, "is_simulation_finished", lambda workspace, metadata: metadata.UniqueName in finished)
    execute_tasks(tasks, SimulationAdapter(tmp_path / "smp", tmp_path), workers=2)
    assert finished == {task.id for task in tasks}
    assert [task.metadata for task in tasks] == originals
