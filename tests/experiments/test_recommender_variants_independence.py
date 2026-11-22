"""Implementation comparisons keep their protocol and authoring tools independent."""

import builtins
import importlib

import pytest

from common.experiments.plans import freeze_plan
from experiments.paper.recommendation import design as stage_design
from experiments.paper.recommender_variants import analysis, design


def test_variant_authoring_imports_do_not_depend_on_intervention_modules(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "experiments.paper.recommendation" or name.startswith("experiments.paper.recommendation."):
            raise AssertionError(f"intervention dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for package, modules in (
        ("recommendation_shared", ("analysis", "plot", "bundle", "runtime")),
        ("recommender_variants", ("design", "analysis", "plot", "variants_review", "study")),
    ):
        for name in modules:
            # Reload also executes import statements when another test already loaded the module.
            importlib.reload(importlib.import_module(f"experiments.paper.{package}.{name}"))


def test_variant_reader_uses_its_own_schema_version(tmp_path, monkeypatch):
    freeze_plan(tmp_path / "plan.json", {
        "schema": design.DESIGN_VERSION, "study": "paper_recommendation",
        "profile": design.STUDY, "protocol": design.design_protocol(),
        "tasks": [design.generate_tasks()[0]],
    })
    monkeypatch.setattr(stage_design, "DESIGN_VERSION", design.DESIGN_VERSION + 1)
    plan, records = analysis.read_study(tmp_path)
    assert plan["schema"] == design.DESIGN_VERSION
    assert records == {}


def test_variant_reader_rejects_intervention_data(tmp_path):
    source = tmp_path / "staged"
    freeze_plan(source / "plan.json", {
        "schema": stage_design.DESIGN_VERSION, "study": "paper_recommendation",
        "profile": "staged", "protocol": {}, "tasks": [stage_design.generate_tasks("staged")[0]],
    })
    with pytest.raises(ValueError, match="unexpected experiment plan"):
        analysis.read_study(source)
