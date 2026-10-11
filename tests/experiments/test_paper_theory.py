import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.paper_theory.micro_validation.population.design import designs as population_designs
from experiments.paper_theory.sweeps import designs
from theory.parameters import Parameters
from theory.runtime.archives import save_result
from theory.runtime.measure import measure_request


def test_scaling_design_keeps_q_over_alpha_and_scaled_horizon():
    tasks = list(designs("scaling", "smoke"))
    assert len(tasks) == 8
    for task in tasks:
        p = Parameters(**task["parameters"])
        request = measure_request(task["id"], p)
        assert request["dynamics"]["rewiring_rate"] / p.alpha == p.ratio
        assert p.horizon <= request["steps"] * p.dt * p.alpha < p.horizon + p.alpha * p.dt + 1e-12
        assert request["dynamics"]["opinion_method"] == "measure"
    with pytest.raises(ValueError, match="exceed one"):
        measure_request("bad", replace(Parameters(), ratio=200))


def test_population_design_separates_initials_from_process_branches():
    tasks = list(population_designs("smoke"))
    assert len(tasks) == 8
    assert {t["parameters"]["degree"] for t in tasks} == {15}
    assert {t["parameters"]["recommendations"] for t in tasks} == {10}
    assert len({t["initial_seed"] for t in tasks}) == 1
    assert len({t["process_seed"] for t in tasks}) == 2


def test_first_step_is_saved_without_fabricating_other_observables(tmp_path):
    item = {"id": "cache", "parameters": {"dt": 1.0, "alpha": 0.1}}
    result = {
        "series": {
            "time": [0.0, 2.0],
            "node_energy": [0.5, 0.1],
            "edge_energy": [1.0, 0.2],
            "polarization": [0.2, 0.8],
        },
        "diagnostics": {},
        "snapshots": {
            "rho": [[0.5, 0.5], [0.5, 0.5]],
            "node_potential": [[1.0, 1.0], [0.8, 0.8]],
            "edge_potential": [[1.0, 1.0], [0.7, 0.7]],
        },
    }
    path = tmp_path / "case.npz"
    save_result(path, item, result)
    with np.load(path, allow_pickle=False) as cache:
        np.testing.assert_allclose(cache["time"], [0, 1, 2])
        np.testing.assert_allclose(cache["node_energy"], [0.5, 0.4, 0.1])
        np.testing.assert_allclose(cache["edge_energy"], [1.0, 0.7, 0.2])
        np.testing.assert_allclose(cache["observable_time"], [0, 2])
        np.testing.assert_allclose(cache["polarization"], [0.2, 0.8])
        assert json.loads(str(cache["metadata"])) == item


def test_parameter_figure_names_preserve_decimal_values(tmp_path):
    import matplotlib.pyplot as plt

    from common.plotting.figures import save

    first, second = tmp_path / "curve_e0.4_r2", tmp_path / "curve_e0.4_r20"
    save(plt.figure(), first)
    save(plt.figure(), second)
    assert {p.name for p in tmp_path.iterdir()} == {
        "curve_e0.4_r2.png",
        "curve_e0.4_r2.pdf",
        "curve_e0.4_r20.png",
        "curve_e0.4_r20.pdf",
    }
