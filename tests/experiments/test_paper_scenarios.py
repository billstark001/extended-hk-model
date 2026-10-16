"""Golden identities from the original ordered paper scenario lists."""

from math import prod

import pytest

from common.io.json import fingerprint
from experiments.paper.scenarios import SCENARIO_AXES, TASKS, generate_tasks


@pytest.mark.parametrize(
    "name,count,identity",
    [
        ("gradation", 12288, "7e973ebb211226ed523e4cb555430337727dc4cc90fbac2c7cdbf1f28ce695ca"),
        ("epsilon", 1600, "b27ac93c0f07483afda9c3d90f8c2d3d4de6156339d55b8e2426f32196f6e772"),
        ("replicate", 200, "a18fd072abbcaf9a5f957d9fc79c45788b5cc572e14a180eef5a57ae6f2872b7"),
        ("mech", 11, "9d9cb0613a526f777b4ea5c0dcd586dde018d994f6be0faf92dd038d7857a600"),
    ],
)
def test_generated_tasks_preserve_every_ordered_metadata_field(name, count, identity):
    tasks = generate_tasks(name)
    assert len(tasks) == count
    assert fingerprint(tasks) == identity
    assert len({t["UniqueName"] for t in tasks}) == count
    assert prod(len(values) for values in SCENARIO_AXES[name].values()) == count
    tasks[0]["HKParams"]["Influence"] = -1
    assert fingerprint(generate_tasks(name)) == identity


def test_task_names_and_axis_order_are_explicit():
    assert TASKS.names == ("epsilon", "gradation", "mech", "replicate")
    assert tuple(SCENARIO_AXES["gradation"]) == ("replicate", "rewiring", "influence", "repost", "recommendation")
    assert tuple(SCENARIO_AXES["epsilon"]) == ("replicate", "tolerance")
    assert tuple(SCENARIO_AXES["replicate"]) == ("replicate", "recommendation")
    assert tuple(SCENARIO_AXES["mech"]) == ("case",)
