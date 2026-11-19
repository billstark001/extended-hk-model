"""Scientific accounting and sampling checks for the manuscript energy audit."""

import numpy as np
import pytest

from common.runtime.failures import ProtocolError
from experiments.paper_theory.evidence.energy_audit import audit_trace, manuscript_source


def accounting_example():
    task = {
        "id": "audit", "source_task_id": "original",
        "parameters": {"alpha": 0.02, "dt": 1},
        "payload": {"request": {"steps": 2}},
    }
    frames = [
        {"step": k, "edge_energy": energy, "node_energy": energy / 2, "budget_rewire": rewiring, "budget_motion": motion}
        for k, energy, rewiring, motion in ((0, 1, 0, 0), (1, 0.92, -0.1, 0.02), (2, 0.86, -0.15, 0.01))
    ]
    result = {"frames": frames, "diagnostics": {"max_node_mass_residual": 1e-15, "max_fixed_degree_residual": 2e-14}}
    source = {"time": np.array([0, 1, 2]), "edge_energy": np.array([1, 0.92, 0.86]), "node_energy": np.array([0.5, 0.46, 0.43])}
    return task, result, source


def test_rewiring_loss_and_positive_cumulative_motion_are_reported_separately():
    task, result, source = accounting_example()
    row = audit_trace(task, result, source, 1e-12)
    assert row["max_abs_budget_residual"] < 1e-15
    assert row["positive_rewiring_increments"] == 0
    assert not row["positive_cumulative_rewiring"]
    assert row["positive_final_motion"] and row["positive_motion_at_any_step"]
    assert row["max_edge_energy_source_prefix_gap"] == 0
    assert row["max_node_energy_source_prefix_gap"] == 0
    assert row["physical_opportunities"] == 2 and row["saved_states"] == 3


def test_unsaved_opportunities_cannot_pass_as_a_full_step_sign_audit():
    task, result, source = accounting_example()
    result["frames"] = result["frames"][::2]
    with pytest.raises(ProtocolError, match="every physical opportunity"):
        audit_trace(task, result, source, 1e-12)


def test_intermediate_budget_failure_is_not_hidden_by_a_closed_final_budget():
    task, result, source = accounting_example()
    result["frames"][1]["edge_energy"] += 0.001
    row = audit_trace(task, result, source, 1e-12)
    assert abs(row["final_budget_residual"]) < 1e-15
    assert row["max_abs_budget_residual"] == pytest.approx(0.001)


def test_positive_rewiring_step_is_detected_even_when_cumulative_rewiring_is_negative():
    task, result, source = accounting_example()
    result["frames"][2].update(budget_rewire=-0.09, edge_energy=0.92, node_energy=0.46)
    row = audit_trace(task, result, source, 1e-12)
    assert row["positive_rewiring_increments"] == 1
    assert not row["positive_cumulative_rewiring"]


def test_nonfinite_budget_cannot_pass_as_a_zero_sign_violation_count():
    task, result, source = accounting_example()
    result["frames"][1]["budget_rewire"] = float("nan")
    with pytest.raises(ProtocolError, match="nonfinite"):
        audit_trace(task, result, source, 1e-12)


def test_changed_reference_trajectory_remains_visible_in_prefix_comparison():
    task, result, source = accounting_example()
    source["edge_energy"][1] += 0.01
    assert audit_trace(task, result, source, 1e-12)["max_edge_energy_source_prefix_gap"] == pytest.approx(0.01)


def test_strict_main_cohort_separates_quadrature_and_step_controls():
    task = {"kind": "measure", "parameters": {"alpha": 0.02, "bins": 161, "degree": 15, "recommendations": 10, "dt": 1, "quadrature": 9}}
    assert manuscript_source(task, strict=False)
    assert not manuscript_source(task)
    task["parameters"].update(quadrature=33, dt=0.5)
    assert manuscript_source(task, strict=False)
    assert not manuscript_source(task)
