"""Partial exports, concurrent execution snapshots and independent CLI phases."""

import csv
import json
import shutil

import pytest

from common.experiments.plans import freeze_plan
from common.io.json import atomic_json, sha
from common.io.records import write_record
from experiments.paper_lifted import __main__ as cli
from experiments.paper_lifted.evidence import analyze as analysis
from experiments.paper_lifted.evidence.design import Design, parameters
from experiments.paper_lifted.evidence.figures import INPUTS, plot
from lifted.runtime.records import record_path


def partial_data(root):
    d = Design("smoke")
    p = parameters(n=40, bins=15)
    initial = d.add("initial", "initial", p)
    atlas = d.add("atlas", "micro_atlas", p, parent=initial, steps=8, observe=(0, 4, 8))
    first = d.add("queries", "micro_probe", p, parent=atlas, at=4, paths=4, steps=24)
    late = d.add("queries", "micro_probe", p, parent=atlas, at=4, paths=4, start=4, steps=24)
    # A completely unexecuted initial state must remain in the ensemble denominator.
    other = d.add("initial", "initial", p, initial=1)
    other_atlas = d.add("atlas", "micro_atlas", p, parent=other, initial=1, steps=8, observe=(0, 4, 8))
    d.add("queries", "micro_probe", p, parent=other_atlas, at=4, initial=1, paths=8, steps=24)
    plan = freeze_plan(root / "plan.json", d.freeze())
    atomic_json(root / "binding.json", {"test": "bound"})
    tasks = {t["id"]: t for t in plan["tasks"]}

    def save(identifier, *, labels=(), states=(), steps=28):
        task = tasks[identifier]
        result = dict(
            states=list(states),
            paths=[
                dict(
                    replicate=i,
                    step=steps,
                    category=label,
                    k_all=1,
                    stop_reason="horizon" if label == "unresolved" else "geometric_terminal",
                )
                for i, label in zip(task["replicates"], labels)
            ],
            files={},
            binding={"test": "bound"},
            dependency_sha256=sha(record_path(root, task["source_task"])) if task["source_task"] else None,
            executed_steps=24,
            elapsed_seconds=0.01,
            wall_seconds=0.02,
            source_physical_hash="physical",
        )
        write_record(record_path(root, identifier), task, result)

    save(initial)
    save(atlas)
    save(first, labels=("k1", "k1", "unresolved", "unresolved"))
    return plan, tasks, save, late


def forbidden(*args, **kwargs):
    raise AssertionError("offline phase attempted model execution")


def test_partial_tables_and_plots_remain_partial_and_track_staleness(tmp_path, monkeypatch):
    plan, tasks, save, late = partial_data(tmp_path)
    monkeypatch.setattr("experiments.paper_lifted.bindings.build", forbidden)
    monkeypatch.setattr("common.runtime.process.run_binary", forbidden)
    assert cli.main(["analyze", "--work-dir", str(tmp_path)]) == 0
    assert cli.main(["plot", "--work-dir", str(tmp_path)]) == 0
    state = analysis.status(tmp_path)
    assert state["analysis_current"] and state["plots_current"] and state["missing"]
    groups = analysis.probability_groups(tmp_path)
    a, b = sorted(groups, key=lambda row: row["initial"])
    assert (a["paths"], a["pending_paths"], a["unresolved"], a["estimate"][0]) == (8, 4, 2, 0.25)
    assert b["pending_paths"] == 8 and b["upper"] == [1] * 4
    with (tmp_path / "analysis/cdf_bounds.csv").open() as stream:
        assert all(int(row["initials"]) == 2 for row in csv.DictReader(stream))
    manifest = json.loads((tmp_path / "analysis/manifest.json").read_text())
    assert not any(name.endswith((".png", ".pdf")) for name in manifest["files"])
    assert not json.loads((tmp_path / "analysis/report.json").read_text())["science_evidence"]
    save(late, labels=("k2",) * 4)
    assert not analysis.status(tmp_path)["analysis_current"]
    assert not analysis.status(tmp_path)["plots_current"]
    analysis.analyze(tmp_path)
    assert analysis.status(tmp_path)["analysis_current"]
    assert not analysis.status(tmp_path)["plots_current"]
    a = min(analysis.probability_groups(tmp_path), key=lambda row: row["initial"])
    assert (a["pending_paths"], a["unresolved"], a["estimate"][:2]) == (0, 2, [0.25, 0.5])
    assert len(tasks) == len(plan["tasks"])


def test_analysis_has_one_snapshot_while_executor_commits_new_records(tmp_path, monkeypatch):
    _, _, save, late = partial_data(tmp_path)
    original = analysis.available
    committed = False

    def concurrent(root, tasks, **kwargs):
        nonlocal committed
        yield from original(root, tasks, **kwargs)
        if not committed:
            committed = True
            save(late, labels=("k2",) * 4)

    monkeypatch.setattr(analysis, "available", concurrent)
    report = analysis.analyze(tmp_path)
    assert report["completed"] == 3
    assert analysis.status(tmp_path)["completed"] == 4
    assert not analysis.status(tmp_path)["analysis_current"]
    assert analysis.analyze(tmp_path)["completed"] == 4
    assert analysis.status(tmp_path)["analysis_current"]


def test_reanalysis_does_not_authenticate_old_plots_and_portable_csv_plot_needs_no_raw(tmp_path, monkeypatch):
    data, portable = tmp_path / "data", tmp_path / "portable"
    partial_data(data)
    analysis.analyze(data)
    first = sha(data / "analysis/manifest.json")
    plot(data / "analysis")
    analysis.analyze(data)
    assert sha(data / "analysis/manifest.json") == first
    portable.mkdir()
    for name in (*INPUTS, "manifest.json"):
        shutil.copyfile(data / "analysis" / name, portable / name)
    monkeypatch.setattr("common.runtime.process.run_binary", forbidden)
    monkeypatch.setattr(analysis, "analyze", forbidden)
    plot(portable)
    assert len(list(portable.glob("*.png"))) == 4
    (portable / "probabilities.csv").write_text("changed")
    with pytest.raises(ValueError, match="tables changed"):
        plot(portable)


def test_phase_analyze_plot_works_without_current_generation_sources(tmp_path, monkeypatch):
    partial_data(tmp_path)
    monkeypatch.setattr("experiments.paper_lifted.run.execute", forbidden)
    monkeypatch.setattr("experiments.paper_lifted.run.sources", lambda: {"different": "code"})
    assert cli.main(["resume", "--work-dir", str(tmp_path), "--phase", "analyze", "--phase", "plot"]) == 0
    assert not (tmp_path / "wall_budget.json").exists()
    assert analysis.status(tmp_path)["plots_current"]


def test_only_run_phase_dispatches_execution(tmp_path, monkeypatch):
    partial_data(tmp_path)
    seen = []

    def execute(args, root, plan):
        seen.append((args.action, root, plan["profile"]))
        return {"missing": ["pending"]}

    monkeypatch.setattr("experiments.paper_lifted.run.execute", execute)
    monkeypatch.setattr(analysis, "analyze", forbidden)
    monkeypatch.setattr("experiments.paper_lifted.evidence.figures.plot", forbidden)
    assert cli.main(["resume", "--work-dir", str(tmp_path), "--phase", "run"]) == 2
    assert seen == [("resume", tmp_path, "smoke")]
    assert not (tmp_path / "analysis").exists()


def test_same_stream_extension_replaces_unknown_path_without_adding_samples(tmp_path):
    d = Design("smoke")
    p = parameters(n=40, bins=15)
    initial = d.add("initial", "initial", p)
    atlas = d.add("atlas", "micro_atlas", p, parent=initial, steps=8, observe=(0, 8))
    probe = d.add("probe", "micro_probe", p, parent=atlas, paths=2, steps=24)
    extension = d.add("tails", "micro_extend", p, parent=probe, paths=1, start=1, steps=24)
    plan = freeze_plan(tmp_path / "plan.json", d.freeze())
    atomic_json(tmp_path / "binding.json", {})
    for task in plan["tasks"]:
        labels = ("k1", "unresolved") if task["id"] == probe else ("k2",) if task["id"] == extension else ()
        write_record(
            record_path(tmp_path, task["id"]),
            task,
            dict(
                binding={},
                files={},
                states=[],
                dependency_sha256=sha(record_path(tmp_path, task["source_task"])) if task["source_task"] else None,
                paths=[
                    dict(replicate=i, step=48 if task["id"] == extension else 24, category=label, stop_reason="horizon")
                    for i, label in zip(task["replicates"], labels)
                ],
            ),
        )
    group = analysis.probability_groups(tmp_path)[0]
    assert group["paths"] == 2 and group["pending_paths"] == group["unresolved"] == 0
    assert group["estimate"] == [0.5, 0.5, 0, 0]


def test_adaptive_query_selection_does_not_change_core_simultaneous_bounds(tmp_path):
    original, formal = tmp_path / "source", tmp_path / "formal"
    plan, _, _, _ = partial_data(original)
    frozen = freeze_plan(formal / "plan.json", {**plan, "profile": "formal"})
    shutil.copytree(original / "raw", formal / "raw")
    shutil.copyfile(original / "binding.json", formal / "binding.json")
    before = analysis.probability_groups(formal)
    extra = Design("formal")
    sample = next(t for t in plan["tasks"] if t["kind"] == "micro_probe")
    extra.add(
        "validation",
        "micro_probe",
        parameters(n=40, bins=15),
        parent=sample["source_task"],
        at=4,
        paths=8,
        seed=123456,
        cohort="mechanism",
    )
    freeze_plan(
        formal / "followup/plan.json",
        {
            **extra.freeze(),
            "parent_plan_fingerprint": frozen["plan_fingerprint"],
            "inputs": {str(p.relative_to(formal)): sha(p) for p in (formal / "raw").glob("*.msgpack.lz4")},
        },
    )
    after = analysis.probability_groups(formal)
    assert len(after) == len(before) + 1
    for prior in before:
        current = next(g for g in after if g["representative_task"] == prior["representative_task"])
        assert current == prior and current["family_size"] == 44


def test_followup_plan_race_cannot_publish_a_mixed_snapshot(tmp_path, monkeypatch):
    partial_data(tmp_path)
    original = analysis.plans

    def selection_during_read(root):
        snapshot = original(root)
        atomic_json(root / "followup/plan.json", {"arrived": "during plan read"})
        return snapshot

    monkeypatch.setattr(analysis, "plans", selection_during_read)
    with pytest.raises(ValueError, match="capturing the analysis snapshot"):
        analysis.analyze(tmp_path)
    assert not (tmp_path / "analysis/manifest.json").exists()
