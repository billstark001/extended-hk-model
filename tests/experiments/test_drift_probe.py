import tempfile
import unittest
from pathlib import Path

import numpy as np

from ehk.plotting.landscape import potential_from_force
from experiments.drift_probe.analyze import load_probe_landscape
from experiments.drift_probe.scenarios import SCENARIO_ORDER, SCENARIOS


class LandscapeUtilityTests(unittest.TestCase):
    def test_potential_integration_supports_multiple_curves(self):
        x = np.array([-1.0, 0.0, 1.0])
        forces = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        potential = potential_from_force(x, forces)
        np.testing.assert_allclose(potential, np.array([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]]))
        np.testing.assert_allclose(np.mean(potential, axis=1), 0.0)

    def test_scenario_order_covers_every_mechanism_map_record(self):
        self.assertEqual(len(SCENARIO_ORDER), 11)
        self.assertEqual(set(SCENARIO_ORDER), set(SCENARIOS))
        self.assertEqual(
            [SCENARIOS[key].unique_name for key in SCENARIO_ORDER],
            [
                "s_mech_baseline",
                "s_mech_baseline_pbs",
                "s_mech_baseline_sbp",
                "s_mech_influence",
                "s_mech_retweet",
                "s_mech_op_recsys",
                "s_mech_phase1",
                "s_mech_phase2",
                "s_mech_phase3",
                "s_mech_phase4",
                "s_mech_phase5",
            ],
        )

    def test_probe_loader_validates_scenario_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counterfactual_probe_baseline.npz"
            np.savez_compressed(
                path,
                scenario_key=np.asarray("pbs"),
                x=np.array([-1.0, 1.0]),
                normalized_time=np.array([0.0]),
                social_force_mean=np.zeros((1, 2)),
                social_force_neighbor_mean=np.zeros((1, 2)),
            )
            with self.assertRaises(ValueError):
                load_probe_landscape("baseline", directory)


if __name__ == "__main__":
    unittest.main()


def test_standalone_probe_uses_shared_selection_plan_and_execution(tmp_path, monkeypatch):
    import json

    from common.experiments.plans import read_plan
    from experiments.drift_probe import run
    from experiments.drift_probe.scenarios import TASKS

    assert tuple(s.key for s in TASKS.select("all")) == SCENARIO_ORDER
    binary = tmp_path / "probe"
    binary.write_bytes(b"fixture")
    output = tmp_path / "result"
    observed = []

    def compute(task):
        observed.append(task)
        path = output / f"counterfactual_probe_{task['id']}.npz"
        path.write_bytes(b"fixture")
        return path, {"key": task["id"], "result": path.name}

    monkeypatch.setattr(run, "probe_task", compute)
    monkeypatch.setattr(run, "run_isolated", lambda f, *args: f(*args))
    run.main(
        [
            "--scenario",
            "pbs",
            "--scenario",
            "sbp",
            "--workspace",
            str(tmp_path),
            "--probe-binary",
            str(binary),
            "--output-dir",
            str(output),
            "--no-plot",
        ]
    )
    assert [t["id"] for t in observed] == ["pbs", "sbp"]
    plan = read_plan(next((output / "runtime").glob("attempt-*/plan.json")))
    assert [t["id"] for t in plan["tasks"]] == ["pbs", "sbp"]
    assert [t["key"] for t in json.loads((output / "manifest.json").read_text())["scenarios"]] == ["pbs", "sbp"]
    status = json.loads(next((output / "runtime").glob("attempt-*/status.json")).read_text())
    assert status["completed"] == 2 and status["missing"] == []
