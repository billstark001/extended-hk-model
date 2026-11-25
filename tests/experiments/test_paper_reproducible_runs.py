"""Seeded execution identities protect historical inputs."""

import json

import pytest

from ehk.micro.simulation import SimulationAdapter
from experiments.paper.scenarios import execution_tasks, generate_tasks
from experiments.paper.study import make_plan


def test_seeded_execution_retains_historical_names_and_has_independent_coordinates():
    old, new = generate_tasks("gradation"), execution_tasks("gradation")
    assert [task["id"] for task in new] == [metadata["UniqueName"] for metadata in old]
    seeds = [json.dumps(task["metadata"]["RNG"], sort_keys=True) for task in new]
    assert len(seeds) == len(set(seeds))
    assert new[0]["coordinates"] == {
        "replicate": 0,
        "rewiring": 0.005,
        "influence": 0.005,
        "repost": 0.0,
        "recommendation": "st",
    }
    assert all("RNG" not in metadata for metadata in old)
    assert execution_tasks("mech") == execution_tasks("mech")
    assert (
        execution_tasks("mech", seed_namespace="independent")[0]["metadata"]["RNG"]
        != execution_tasks("mech")[0]["metadata"]["RNG"]
    )
    assert json.loads(json.dumps(make_plan("mech"))) == make_plan("mech")

def test_seeded_runner_cannot_cache_unseeded_or_differently_seeded_output(tmp_path):
    task = execution_tasks("mech")[0]
    directory = tmp_path / task["id"]
    directory.mkdir()
    (directory / "finished-000001.msgpack").touch()
    path = directory / "metadata.json"
    path.write_text(json.dumps(generate_tasks("mech")[0]))
    adapter = SimulationAdapter(tmp_path / "smp", tmp_path, strict_metadata=True)
    with pytest.raises(ValueError, match="existing simulation differs.*RNG"):
        adapter.is_cached(task)
    actual = {**task["metadata"], "runtime_optional_field": 1}
    actual.pop("ViewPostsDetails")  # SMP omits this optional false value.
    path.write_text(json.dumps(actual))
    assert adapter.is_cached(task)
    actual["RNG"] = {**actual["RNG"], "Seed1": "0x0000000000000000"}
    path.write_text(json.dumps(actual))
    with pytest.raises(ValueError, match="existing simulation differs in RNG"):
        adapter.is_cached(task)

def test_original_run_plan_prints_without_resolving_private_workspace(capsys):
    from experiments.paper.run import main

    main(["mech", "--plan-only"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["schema"] == 2 and plan["tasks"][0]["metadata"]["RNG"]
