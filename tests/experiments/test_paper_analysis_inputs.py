"""Explicit scientific input declarations and side-effect-free selections."""

from dataclasses import replace

import pytest

from experiments.paper.analysis import tasks


def test_plan_selection_creates_no_output_and_rejects_unknown_tasks(tmp_path):
    database = tmp_path / "not-created" / "stats.sqlite"
    name = "s_grad_sim0_rw0_dc0_rt0_st"
    plan = tasks.run_mode(
        "gradation", None, 1, database=database, raw_root=tmp_path / "missing",
        plan_only=True, task_names=[name], metadata_source="original-design",
    )
    assert plan.planned == 1
    assert plan.tasks[0].UniqueName == name
    assert not plan.workspace_available
    assert not database.parent.exists()
    with pytest.raises(ValueError, match="unknown"):
        tasks.run_mode("gradation", None, 1, raw_root=tmp_path, task_names=["not-a-task"], plan_only=True)


def test_missing_recorded_metadata_requires_explicit_design_declaration(tmp_path, monkeypatch):
    import smp_bindings

    metadata = tasks.cfg.generate_tasks("gradation")[0]
    (tmp_path / metadata.UniqueName).mkdir()

    class Raw:
        is_finished = is_sanitized = True
        agents = 500

        def __init__(self, directory, metadata):
            self.metadata = metadata

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(smp_bindings, "RawSimulationRecord", Raw)
    monkeypatch.setitem(
        tasks.MODE_CONFIGS, "gradation",
        replace(tasks.MODE_CONFIGS["gradation"], common_stats=[], include_backdrop=False),
    )
    with pytest.raises(FileNotFoundError):
        tasks.get_statistics_for_mode(metadata, str(tmp_path), "fixture", None, "gradation")
    row = tasks.get_statistics_for_mode(
        metadata, str(tmp_path), "fixture", None, "gradation", metadata_source="original-design",
    )
    assert row.recorded_metadata is None
    assert row.metadata_source == "original-design"
    assert "not independently recorded" in row.metadata_provenance
    assert row.rng_provenance == "unavailable in raw metadata"
    assert row.metadata.RNG is None and row.metadata.RNGSeed is None

