"""Named tasks must complete their own plan/analysis/plot cycle in isolation."""

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from common.io.json import atomic_json, sha
from common.io.records import write_record
from experiments.paper_theory import __main__ as cli
from experiments.paper_theory.evidence.design import PROFILES, canonical_profile, design
from experiments.paper_theory.evidence.main_design import TOPICS
from theory.runtime.archives import save_result


@pytest.mark.parametrize("prefix", ["main", "main_smoke"])
def test_aggregate_is_the_deduplicated_union_of_named_topics(prefix):
    aggregate = design(prefix)
    combined = {}
    for topic in TOPICS:
        for task in design(f"{'smoke_' if prefix == 'main_smoke' else ''}{topic}")["tasks"]:
            spec = {k: v for k, v in task.items() if k != "roles"}
            if task["id"] in combined:
                assert combined[task["id"]] == spec
            combined[task["id"]] = spec
    assert combined == {t["id"]: {k: v for k, v in t.items() if k != "roles"} for t in aggregate["tasks"]}
    assert "closure" not in {r for t in aggregate["tasks"] for r in t["roles"]}
    if prefix == "main":
        assert Counter(t["kind"] for t in aggregate["tasks"]) == {"measure": 738, "micro": 1056}


LEGACY_PROFILES = tuple(f"main_{t}" for t in TOPICS) + tuple(f"main_smoke_{t}" for t in TOPICS)


def synthetic_records(directory, plan):
    """Small deterministic fixtures exercise consumers without a scientific run."""
    for task in plan["tasks"]:
        p = task["parameters"]
        count = round(p["horizon"] / (p["alpha"] * p["dt"]))
        time = np.arange(count + 1) * p["dt"]
        tau = time * p["alpha"]
        rho = np.full(p["bins"], 1 / p["bins"])
        edge = p["degree"] * np.outer(rho, rho).ravel()
        node = np.exp(-(1 + p["alpha"]) * tau)
        link = np.exp(-(1 + p["ratio"] / 5) * tau)
        diagnostics = {
            "converged": False,
            "max_node_mass_residual": 0,
            "max_fixed_degree_residual": 0,
            "executed_steps": count,
        }
        if task["kind"] == "measure":
            save_result(
                directory / "measure" / "raw" / (task["id"] + ".npz"),
                task,
                {
                    "series": {"time": time, "node_energy": node, "edge_energy": link},
                    "diagnostics": diagnostics,
                    "snapshots": {
                        "time": time,
                        "rho": np.tile(rho, (len(time), 1)),
                        "edge": np.tile(edge, (len(time), 1)),
                        "final_rho": rho,
                        "final_edge": edge,
                    },
                },
            )
        else:
            pair = [
                {
                    "time": float(t),
                    "step": i,
                    "node_energy": float(n),
                    "edge_energy": float(e),
                    "rho": rho,
                    "edge": edge,
                    "variance": 0.2,
                    "discordant": 0.3,
                }
                for i, (t, n, e) in enumerate(zip(time, node, link))
            ]
            micro = [
                {
                    **f,
                    "exact_node_energy": float(np.exp(-(1 + task["branch"] / 10) * tau[i])),
                    "exact_edge_energy": float(link[i]),
                    "exact_variance": 0.2,
                    "exact_discordant": 0.3,
                }
                for i, f in enumerate(pair)
            ]
            write_record(
                directory / "micro" / "raw" / (task["id"] + ".msgpack.lz4"),
                task,
                {
                    "initial_hash": f"n{p['population']}_i{task['initial_seed']}",
                    "frames": micro,
                    "pair": {"frames": pair, "diagnostics": diagnostics},
                },
            )
    for kind in {t["kind"] for t in plan["tasks"]}:
        atomic_json(directory / kind / "binding.json", {"binary_sha256": "fixture", "request_schema_sha256": "fixture"})


@pytest.mark.parametrize("topic", TOPICS)
def test_topic_works_without_other_topics_or_runtime(tmp_path, topic):
    profile = f"smoke_{topic}"
    assert profile in PROFILES
    plan = cli.main(["plan", "--profile", profile, "--work-dir", str(tmp_path)])
    directory = tmp_path / "evidence" / profile
    assert len(list(directory.glob("*/design.json"))) == 1
    assert cli.main(["plan", "--profile", profile, "--work-dir", str(tmp_path)]) == plan
    synthetic_records(directory, plan)
    report = cli.main(["analyze", "--run-dir", str(tmp_path)])
    assert report["missing"] == []
    assert report["completed"] == len(plan["tasks"])
    cli.main(["plot", "--run-dir", str(tmp_path), "--output-dir", str(tmp_path / "raw_figures")])
    cli.main(["plot", "--run-dir", str(tmp_path), "--tables-only", "--output-dir", str(tmp_path / "csv_figures")])
    assert list((tmp_path / "raw_figures").glob("*.pdf"))
    assert list((tmp_path / "csv_figures").glob("*.pdf"))
    manifest = json.loads((directory / "figure_manifest.json").read_text())
    assert manifest["research_topics"] == [topic]
    status = cli.main(["status", "--run-dir", str(tmp_path)])
    assert status["raw_available"]
    assert status["valid_raw_records"] == len(plan["tasks"])


@pytest.mark.parametrize("topic,kind", [("scaling", "measure"), ("matched", "micro")])
def test_run_dispatch_calls_only_the_selected_model(tmp_path, monkeypatch, topic, kind):
    from experiments.paper_theory import study
    from experiments.paper_theory.study import TheoryStudy

    binary = tmp_path / "fixture-binary"
    binary.write_text("not executed")
    build = tmp_path / "build"
    atomic_json(build / "manifest.json", {"binary": str(binary), "binary_sha256": sha(binary)})
    seen = []

    def execute(selected, *args, **kwargs):
        assert {t["kind"] for t in selected} == {kind}
        seen.extend(t["id"] for t in selected)

    def forbidden(*args, **kwargs):
        raise AssertionError("the other model must not execute or bind an empty design")

    monkeypatch.setattr(study, "execute_tasks", execute)
    monkeypatch.setattr(study, "MeasureAdapter", (lambda *a, **kw: object()) if kind == "measure" else forbidden)
    monkeypatch.setattr(study, "PopulationAdapter", (lambda *a, **kw: object()) if kind == "micro" else forbidden)
    args = SimpleNamespace(
        action="run",
        profile=f"smoke_{topic}",
        horizon=None,
        binary=binary,
        engine=binary,
        force=False,
        limit=None,
        workers=1,
    )
    TheoryStudy(args, tmp_path, build, None, None).workflow().dispatch("run")
    assert seen == [t["id"] for t in design(args.profile)["tasks"]]


def test_retired_migration_command_is_unavailable(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["migrate", "--run-dir", str(tmp_path)])


@pytest.mark.parametrize("legacy", LEGACY_PROFILES)
def test_legacy_names_read_frozen_designs_but_are_not_new_cli_choices(legacy):
    current = design(canonical_profile(legacy))
    from experiments.paper_theory.evidence.design import topics

    assert topics(legacy) == topics(current["profile"])
    with pytest.raises(ValueError):
        design(legacy)
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["plan", "--profile", legacy])
