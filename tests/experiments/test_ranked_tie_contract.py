"""New plans identify the fixed tie rule without relabeling historical evidence."""

from dataclasses import replace

import pytest

from common.io.json import atomic_json, fingerprint
from common.runtime.failures import ProtocolError
from ehk.domain import RANKED_TIE_RULE, StageGate, freeze_plan, read_plan, to_wire
from experiments.paper import study as original
from experiments.paper.pathway_controls.design import make_plan as control_plan
from experiments.paper.recommendation import analysis, study
from experiments.paper.recommendation_shared import runtime
from experiments.paper.recommender_variants.study import make_plan as variant_plan
from experiments.paper.recommender_variants import study as variants


@pytest.mark.parametrize("factory", (
    lambda: original.make_plan("mech"),
    lambda: control_plan(replicates=1, cases=("op_recsys",), horizon=2, node_count=8, follow_count=2),
    lambda: study.make_plan("staged"),
    lambda: study.make_plan("staged_timing"),
    variant_plan,
))
def test_current_contract_changes_plan_identity_without_relabeling_old_wire(tmp_path, factory):
    plan = factory()
    assert plan.protocol.ranked_tie_rule == RANKED_TIE_RULE
    plan = replace(plan, tasks=plan.tasks[:1])
    historical = to_wire(plan)
    historical["protocol"].pop("ranked_tie_rule")
    historical["plan_fingerprint"] = fingerprint(historical)
    path = tmp_path / "plan.json"
    atomic_json(path, historical)

    saved = read_plan(path)
    assert saved.protocol.ranked_tie_rule is None
    assert saved.tasks == plan.tasks
    assert to_wire(saved) == historical
    assert saved.plan_fingerprint == historical["plan_fingerprint"]
    with pytest.raises(ProtocolError, match="plan changed.*new output directory"):
        freeze_plan(path, plan)


@pytest.mark.parametrize("recorded_rule", (None, "id-order-v1", "different-rule"))
def test_timing_rejects_unknown_or_different_tie_evidence_before_gate(tmp_path, monkeypatch, recorded_rule):
    source = study.make_plan("staged")
    source = replace(source, protocol=replace(source.protocol, ranked_tie_rule=recorded_rule))
    monkeypatch.setattr(analysis, "read_study", lambda *args: (source, {}))
    monkeypatch.setattr(analysis, "stage_gate", lambda *args: pytest.fail("unmatched evidence reached gate"))
    monkeypatch.setattr(runtime, "run_dataset", lambda *args, **kwargs: pytest.fail("unmatched evidence started execution"))
    with pytest.raises(ValueError, match="different or unrecorded ranked tie rule"):
        study.run_study("staged_timing", tmp_path / "timing", tmp_path / "smp", 1,
                        stage_evidence=tmp_path / "staged")


def test_timing_records_current_rule_gate_provenance_without_running(tmp_path, monkeypatch):
    source = replace(study.make_plan("staged"), plan_fingerprint="current-rule-source")
    gate = StageGate(n=2, mean=.1, lo=.05, hi=.15, planned_core=2, completed_core=2,
                     complete=True, passed=True, minimum_effect=.05)
    monkeypatch.setattr(analysis, "read_study", lambda *args: (source, {}))
    monkeypatch.setattr(analysis, "stage_gate", lambda *args: gate)

    def inspect_binding(*args, **kwargs):
        plan = kwargs["plan_factory"]()
        binding = kwargs["binding_factory"](plan, tmp_path / "binding.json", "b" * 64)
        assert plan.protocol.ranked_tie_rule == RANKED_TIE_RULE
        assert binding.stage_evidence.plan == source.plan_fingerprint
        assert binding.stage_evidence.contrast == gate
        return "not-executed"

    monkeypatch.setattr(runtime, "run_dataset", inspect_binding)
    assert study.run_study("staged_timing", tmp_path / "timing", tmp_path / "smp", 1) == "not-executed"


def test_invalid_arm_binding_requests_a_new_dataset_without_rewriting_it(tmp_path):
    path = tmp_path / "binding.json"
    atomic_json(path, {"binary_sha256": "historical-producer"})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="preserve the original binding.*new dataset directory"):
        variants._binding(variant_plan(), path, "corrected-producer")
    assert path.read_bytes() == before
