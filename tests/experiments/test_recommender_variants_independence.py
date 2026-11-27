"""Implementation comparisons keep their protocol and authoring tools independent."""

import builtins
import importlib

import pytest

from ehk.domain import freeze_plan
from common.io.json import atomic_json
from experiments.paper.recommendation_shared.storage import StudyStore
from experiments.paper.recommendation import design as stage_design
from experiments.paper.recommender_variants import analysis, design
from tests.experiments.recommendation_fixtures import plan as fixture_plan


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
    freeze_plan(tmp_path / "plan.json", fixture_plan("recommender_variants", tasks=design.generate_tasks()[:1]))
    atomic_json(tmp_path / "runtime/binding.json", {"binary_sha256": "schema-fixture"})
    with StudyStore(tmp_path):
        pass
    monkeypatch.setattr(stage_design, "DESIGN_VERSION", design.DESIGN_VERSION + 1)
    plan, records = analysis.read_study(tmp_path)
    assert plan.schema == design.DESIGN_VERSION
    assert records == {}


def test_variant_reader_rejects_intervention_data(tmp_path):
    source = tmp_path / "staged"
    freeze_plan(source / "plan.json", fixture_plan(tasks=stage_design.generate_tasks("staged")[:1]))
    with pytest.raises(ValueError, match="unexpected experiment plan"):
        analysis.read_study(source)
