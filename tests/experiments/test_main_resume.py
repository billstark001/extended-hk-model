"""Interrupted task recovery, provenance checks and exclusive full-run execution."""

import json
import os
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from common.io.json import atomic_json, sha
from common.runtime.process import run_jsonl, run_tasks
from experiments.paper_theory.evidence.resume import recover_task_lock, restore_binaries, session_lock, settings
from experiments.paper_theory.micro_validation.population import run as population
from experiments.paper_theory.runtime import measure_runs
from theory.parameters import Parameters


def task(identifier):
    return {
        "id": identifier,
        "parameters": asdict(Parameters(alpha=0.02, horizon=0.02)),
        "initial_seed": 1,
        "process_seed": 2,
        "selection": "random",
    }


def diagnostics():
    return {"max_node_mass_residual": 0, "max_fixed_degree_residual": 0, "executed_steps": 1}


def measure_result():
    return {
        "series": {"time": [0.0, 1.0], "node_energy": [1.0, 0.0], "edge_energy": [1.0, 0.0]},
        "diagnostics": diagnostics(),
        "snapshots": {"final_rho": [1.0], "final_edge": [1.0]},
    }


def test_cancellable_kinetic_transport_preserves_binding_request_id_protocol(monkeypatch):
    def transport(binary, line, *, arguments):
        assert binary == "kinetic" and arguments == ("batch",)
        assert json.loads(line)["request_id"] == "one"
        return json.dumps({"request_id": "one", "result": measure_result()})

    monkeypatch.setattr(measure_runs, "run_jsonl", transport)
    assert measure_runs.simulate("kinetic", "one", Parameters())["diagnostics"] == diagnostics()
    monkeypatch.setattr(measure_runs, "run_jsonl", lambda *a, **kw: json.dumps({"request_id": "other", "result": {}}))
    with pytest.raises(RuntimeError, match="invalid kinetic response"):
        measure_runs.simulate("kinetic", "one", Parameters())


def test_measure_interrupt_then_resume_skips_completed_tasks(tmp_path, monkeypatch):
    binary = tmp_path / "binary"
    binary.write_bytes(b"original")
    tasks = [task(str(i)) for i in range(3)]
    seen = []

    def simulate(binary, identifier, parameters, **options):
        seen.append(identifier)
        if identifier == "1":
            raise KeyboardInterrupt()
        return measure_result()

    monkeypatch.setattr(measure_runs, "simulate", simulate)
    output = tmp_path / "measure"
    with pytest.raises(KeyboardInterrupt):
        measure_runs.execute(tasks, output, binary)
    assert seen == ["0", "1"]
    complete = output / "raw" / "0.npz"
    original, modified = complete.read_bytes(), complete.stat().st_mtime_ns
    # A truncated final record and an interrupted temporary write both need work.
    (output / "raw" / "1.npz").write_bytes(b"incomplete ZIP")
    (output / "raw" / "2.tmp").write_bytes(b"interrupted temporary record")
    seen.clear()
    monkeypatch.setattr(measure_runs, "simulate", lambda b, i, p, **kw: seen.append(i) or measure_result())
    measure_runs.execute(tasks, output, binary, workers=2)
    assert sorted(seen) == ["1", "2"]
    assert complete.read_bytes() == original and complete.stat().st_mtime_ns == modified
    assert len(list((output / "raw").glob("*.npz"))) == 3
    # Different binaries must fail before any result can be reused or overwritten.
    binary.write_bytes(b"changed")
    with pytest.raises(ValueError, match="binary changed"):
        measure_runs.execute(tasks, output, binary)


def test_population_resume_recomputes_truncated_result_but_rejects_other_protocol(tmp_path, monkeypatch):
    binary = tmp_path / "binary"
    binary.write_bytes(b"original")
    tasks = [task(str(i)) for i in range(2)]
    calls = []

    def respond(binary, line):
        identifier = json.loads(line)["id"]
        calls.append(identifier)
        frames = [
            {"time": 0.0, "node_energy": 1.0, "edge_energy": 1.0},
            {"time": 1.0, "node_energy": 0.0, "edge_energy": 0.0},
        ]
        return json.dumps(
            {
                "id": identifier,
                "status": "ok",
                "result": {"frames": frames, "pair": {"frames": frames, "diagnostics": diagnostics()}},
            }
        )

    monkeypatch.setattr(population, "run_jsonl", respond)
    raw = tmp_path / "micro" / "raw"
    population.execute(tasks, raw, binary)
    first = raw / "0.json"
    original, modified = first.read_bytes(), first.stat().st_mtime_ns
    (raw / "1.json").write_text('{"task":')
    calls.clear()
    population.execute(tasks, raw, binary)
    assert calls == ["1"]
    assert first.read_bytes() == original and first.stat().st_mtime_ns == modified
    changed = {**tasks[0], "process_seed": 999}
    with pytest.raises(ValueError, match="protocol differs"):
        population.execute([changed], raw, binary)
    assert first.read_bytes() == original


def test_task_failure_kills_active_children_and_never_launches_remaining_tasks(tmp_path):
    pid_path = tmp_path / "child.pid"
    seen = []

    def run(index):
        seen.append(index)
        if index == 0:
            code = f"import os,time; open({str(pid_path)!r},'w').write(str(os.getpid())); time.sleep(30)"
            run_jsonl(sys.executable, "", arguments=("-c", code))
        else:
            deadline = time.monotonic() + 5
            while not pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            raise RuntimeError("interrupted task")

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="interrupted task"):
        run_tasks(run, [0, 1, 2], 2)
    assert time.monotonic() - started < 5
    assert sorted(seen) == [0, 1]
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)


def test_session_lock_prevents_duplicate_launch_and_is_released_on_failure(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with session_lock(tmp_path):
            with pytest.raises(RuntimeError, match="already active"):
                with session_lock(tmp_path):
                    pytest.fail("second launcher acquired lock")
            raise KeyboardInterrupt()
    with session_lock(tmp_path):
        pass


def test_stale_task_lock_is_recovered_but_live_owner_is_preserved(tmp_path, monkeypatch):
    lock = tmp_path / "active.lock"
    lock.write_text(str(os.getpid()))
    with pytest.raises(RuntimeError, match="already active"):
        recover_task_lock(tmp_path)
    assert lock.exists()
    monkeypatch.setattr(os, "kill", lambda *args: (_ for _ in ()).throw(ProcessLookupError()))
    recover_task_lock(tmp_path)
    assert not lock.exists()


def config_fixture(tmp_path):
    for name in ("meso", "micro"):
        root = tmp_path / name
        root.mkdir()
        (root / "go.mod").write_text(f"module {name}\n")
        (root / "go.sum").write_text("")
        (root / "main.go").write_text("package main\n")
    work = tmp_path / "work"
    work.mkdir()
    args = SimpleNamespace(
        resume=False,
        profile="smoke",
        horizon=None,
        backend="purego",
        meso_root=tmp_path / "meso",
        micro_root=tmp_path / "micro",
    )
    config = settings(work, tmp_path / "publish", args)
    args.resume = True
    return work, args, config


@pytest.mark.parametrize("changed", ["profile", "horizon", "backend", "source", "schema"])
def test_resume_rejects_changed_scientific_inputs(tmp_path, changed):
    work, args, config = config_fixture(tmp_path)
    original = (work / "run_manifest.json").read_bytes()
    if changed == "source":
        (args.meso_root / "main.go").write_text("package changed\n")
    elif changed == "schema":
        config["request_schema_sha256"] = "wrong"
        atomic_json(work / "run_manifest.json", config)
        original = (work / "run_manifest.json").read_bytes()
    else:
        setattr(args, changed, {"profile": "paper", "backend": "accelerate", "horizon": 1.0}[changed])
    with pytest.raises(ValueError, match="changed|cannot change"):
        settings(work, tmp_path / "publish", args)
    assert (work / "run_manifest.json").read_bytes() == original


def test_original_binaries_restored_without_rebuild_and_tampering_rejected(tmp_path):
    work, args, config = config_fixture(tmp_path)
    for name, folder in (("paper-theory-engine", "build"), ("smp-kinetic", "bin")):
        directory = work / folder
        directory.mkdir()
        binary = directory / name
        binary.write_bytes(name.encode())
        manifest = {"binary_sha256": sha(binary), "backend": config["backend"]}
        if folder == "build":
            manifest.update(config["sources"])
        else:
            manifest.update(source=config["sources"]["meso"], adapter_source=config["sources"]["engine"])
        atomic_json(directory / ("manifest.json" if folder == "build" else "kinetic_manifest.json"), manifest)
    stage = tmp_path / "new-stage"
    stage.mkdir()
    restored = restore_binaries(work, stage, config)
    assert [p.name for p in restored] == ["paper-theory-engine", "smp-kinetic"]
    assert restored[1].read_bytes() == b"smp-kinetic"
    (work / "bin" / "smp-kinetic").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        restore_binaries(work, stage, config)


def test_build_interruption_can_restart_but_bound_tasks_require_original_binary(tmp_path):
    work, args, config = config_fixture(tmp_path)
    assert restore_binaries(work, tmp_path, config) is None
    binding = work / "evidence" / "main_smoke" / "micro" / "binding.json"
    atomic_json(binding, {})
    with pytest.raises(ValueError, match="original executables"):
        restore_binaries(work, tmp_path, config)
