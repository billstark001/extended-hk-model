"""Retain empirical energy fits and single-slot figures independently of scans."""

import csv
import json

import numpy as np
import pytest

from common.io.json import sha
from experiments.paper_theory import __main__ as cli
from experiments.paper_theory.energy_scan.expanded import config
from experiments.paper_theory.energy_scan.expanded.fit.edge import fit_edge_timescales, fit_slot
from experiments.paper_theory.energy_scan.expanded.fit.node import low_parameter_fits
from experiments.paper_theory.energy_scan.expanded.storage import load_shard, load_slot
from theory.release.models import times_node


def archive_fixture(path, *, baseline=False, coarse=False):
    tau = np.unique(np.r_[0, 0.01, np.linspace(0.1, 100, 501)])
    node = np.exp(-0.3 * tau)
    edge = 0.65 * np.exp(-((0.9 * 2 * tau) ** 1.3)) + 0.35 * node
    delta = np.array([node[1] - node[0], edge[1] - edge[0]])
    if coarse:
        tau, node, edge = (np.delete(a, 1) for a in (tau, node, edge))
    shape = (2, 2, len(tau))
    length = np.zeros((2, 2), dtype=int)
    length[1, 1] = len(tau)
    curves = {k: np.full(shape, np.nan) for k in ("time" if baseline else "tau", "node_energy", "edge_energy")}
    for name, values in zip(curves, (tau / 0.02 if baseline else tau, node, edge)):
        curves[name][1, 1] = values
    first = np.zeros((2, 2, 2))
    first[1, 1] = delta
    np.savez_compressed(
        path,
        protocol=json.dumps({"dynamics": "hk", "alpha": 0.02, "dt": 0.5}),
        epsilon=np.array([0.2, 0.4]),
        q_over_alpha=np.array([0, 2]),
        lengths=length,
        converged=np.ones((2, 2), dtype=bool),
        first_step_delta=first,
        **curves,
    )


@pytest.mark.parametrize("baseline,coarse", [(False, False), (False, True), (True, False)])
def test_select_one_slot_and_preserve_scaled_time_and_physical_first_update(tmp_path, baseline, coarse):
    source = tmp_path / "hk_alpha_0.02.npz"
    archive_fixture(source, baseline=baseline, coarse=coarse)
    original = sha(source)
    slot, path = load_slot(source, dynamics="hk", alpha=0.02, epsilon=0.4, ratio=2)
    assert path == source
    assert slot["tau"][1] == 0.01  # Physical dt=.5, not the retired design's dt=1.
    assert slot["tau"][-1] == 100
    assert slot["node_energy"][1] == pytest.approx(np.exp(-0.003))
    assert sha(source) == original
    assert load_slot(tmp_path, dynamics="hk", alpha=0.02, epsilon=0.4, ratio=2)[0]["q"] == 0.04
    with pytest.raises(ValueError, match="saved protocol"):
        load_slot(source, dynamics="deffuant", alpha=0.02, epsilon=0.4, ratio=2)
    with pytest.raises(ValueError, match="exactly one"):
        load_slot(source, dynamics="hk", alpha=0.02, epsilon=0.4, ratio=3)


def test_original_node_quantile_models_remain_available_without_grid():
    times = times_node(config.PROGRESS_LEVELS, 0.1, 0.3)
    row = dict(dynamics="hk", epsilon=0.4, alpha=0.02, q=0.04, q_over_alpha=2, energy="node_energy")
    row.update({f"tau_{round(100 * p):02d}": t for p, t in zip(config.PROGRESS_LEVELS, times)})
    fits = low_parameter_fits([row])
    assert {r["model"] for r in fits} == {"gompertz_hazard_2", "exponentiated_weibull_3", "rational_quadratic_bezier_3"}
    gompertz = next(r for r in fits if r["model"] == "gompertz_hazard_2")
    assert gompertz["initial_hazard"] == pytest.approx(0.1, rel=1e-6)
    assert gompertz["acceleration"] == pytest.approx(0.3, rel=1e-6)
    assert gompertz["maximum_absolute_relative_error"] < 1e-6


def test_edge_fit_recovers_mixture_and_zero_rewiring_has_no_fast_shape(tmp_path):
    source = tmp_path / "hk_alpha_0.02.npz"
    archive_fixture(source)
    slot, _ = load_slot(source, dynamics="hk", alpha=0.02, epsilon=0.4, ratio=2)
    fit = fit_slot(slot)
    assert fit["weight"] == pytest.approx(0.65, rel=1e-5)
    assert fit["kappa"] == pytest.approx(0.9, rel=1e-5)
    assert fit["beta"] == pytest.approx(1.3, rel=1e-5)
    assert fit["rms_normalized_energy_error"] < 1e-7
    zero = fit_slot({**slot, "q_over_alpha": 0, "q": 0})
    assert zero["weight"] == 0
    assert np.isnan(zero["kappa"]) and np.isnan(zero["beta"])
    assert not zero["shape_identifiable"]
    assert fit_slot({**slot, "node_energy": np.ones_like(slot["tau"])}) is None
    shard = load_shard(source)
    # Make an actual one-slot shard; fit_edge_timescales never requests absent grid cells.
    shard = {
        "protocol": shard["protocol"],
        "epsilon": np.array([0.4]),
        "q_over_alpha": np.array([2]),
        "lengths": np.array([[len(slot["tau"])]]),
        "converged": np.array([[True]]),
        **{k: slot[k][None, None, :] for k in ("tau", "node_energy", "edge_energy", "first_step_delta")},
    }
    assert len(fit_edge_timescales({("hk", 0.02): shard})) == 1


def test_cli_exports_only_selected_slot_fits_and_figure(tmp_path):
    source = tmp_path / "hk_alpha_0.02.npz"
    archive_fixture(source, coarse=True)
    before = sha(source)
    args = ["--run-dir", str(source), "--dynamics", "hk", "--epsilon", "0.4", "--alpha", "0.02", "--ratio", "2"]
    output = tmp_path / "fits"
    manifest = cli.main(["fit-slot", *args, "--output-dir", str(output)])
    assert manifest["figure"] is None
    with (output / "low_parameter_fits.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6 and {r["energy"] for r in rows} == {"node_energy", "edge_energy"}
    manifest = cli.main(["plot-slot", *args, "--output-dir", str(output)])
    assert manifest["source_sha256"] == before == sha(source)
    assert manifest["slot"]["epsilon"] == 0.4
    assert {p.suffix for p in output.glob(manifest["figure"] + ".*")} == {".png", ".pdf"}
    assert len(list(output.glob("*.csv"))) == 3
    with pytest.raises(SystemExit):
        cli.main(["plot-slot", "--run-dir", str(source)])
