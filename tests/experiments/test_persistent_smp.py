"""Persistent process transport keeps resumable history on cancellation."""

import json
import sys
import threading
import time
from pathlib import Path
from dataclasses import replace
from copy import deepcopy
from ehk.domain import PolicyChange, to_wire
from ehk.micro.scenarios import create_scenario_metadata

import pytest

from common.runtime.failures import ProtocolError
from ehk.micro.persistent import observe_persistent


@pytest.fixture
def fake_smp(tmp_path, monkeypatch):
    import subprocess
    original_popen = subprocess.Popen
    monkeypatch.setattr("ehk.micro.persistent.subprocess.Popen", lambda argv, **kwargs: original_popen([sys.executable, *argv], **kwargs))
    binary = tmp_path / "smp"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import gzip, json, signal, sys, time\n"
        "from pathlib import Path\n"
        "assert sys.argv[1] == '--job'\n"
        "job = json.loads(Path(sys.argv[2]).read_text())\n"
        "run = Path(job['base_path']) / job['metadata']['UniqueName']\n"
        "checkpoint = run / 'resumable'\n"
        "if job['metadata']['UniqueName'] == 'resume' and not checkpoint.exists():\n"
        "    def stop(signum, frame):\n"
        "        checkpoint.write_text('saved')\n"
        "        sys.exit(130)\n"
        "    (run / 'ready').touch()\n"
        "    while True:\n"
        "        if Path(sys.argv[2] + '.cancel').exists(): stop(None, None)\n"
        "        time.sleep(0.01)\n"
        "steps = job['observe_at']\n"
        "n = job['metadata']['NodeCount']\n"
        "points = []\n"
        "previous = steps[0]\n"
        "for step in steps:\n"
        "    slot = dict(shown=0,cross_camp=0,cross_confident=0,cross_rewire=0)\n"
        "    exp = dict(updates=(step-previous)*n,requested=2*(step-previous)*n,neighbor_seen=0,neighbor_concordant=0,neighbor_with_posts=0,neighbor_share_sum=0.0,recommended=2*(step-previous)*n,exploration=slot,policy=slot,bridge=slot,cross_camp=0,cross_confident=0,cross_rewire=0,rewires=0,absolute_shift=0.0)\n"
        "    pathway = dict(separation=0.0,pair_concordance=1.0,edge_concordance=1.0,excess_concordance=0.0)\n"
        "    points.append({'step':step,'relative_step':step-steps[0],'exposure':exp,'pathway':pathway})\n"
        "    previous = step\n"
        "terminal=dict(status='absorbed',category='consensus',k_all=1,k_major=1,components=[],margins=dict(gap_to_epsilon=0.0,diameter_to_epsilon=0.0,mass_to_major_cutoff=0.0))\n"
        "result = {'endpoints':points,'terminal':terminal,'stop_reason':'horizon','stopped_at':steps[-1],'final_opinions':[0.0]*n}\n"
        "if 'exposure_trace' in job:\n"
        "    path = Path(job['exposure_trace'])\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    with gzip.open(path, 'wt') as trace: trace.write('{}\\n')\n"
        "    result['exposure_trace'] = {'version': 1, 'path': str(path), 'updates': (steps[-1] - steps[0]) * n, 'slots': 2 * (steps[-1] - steps[0]) * n}\n"
        "(run / 'trajectory.json').write_text('{}')\n"
        "(run / 'finished-000001.msgpack').touch()\n"
        "print(json.dumps({'type': 'research_result', 'protocol': job['result_protocol'], 'result': result}))\n"
    )
    binary.chmod(0o755)
    return binary


def test_persistent_adapter_freezes_the_executable_job(fake_smp, tmp_path):
    base = tmp_path / "raw"
    metadata = replace(create_scenario_metadata("paired-arm"),NodeCount=3)
    original_metadata=deepcopy(metadata)
    change = PolicyChange(at=2,recommender="StructureM9")
    result = observe_persistent(fake_smp, base, metadata, [0, 2, 5], policy_change=change)
    assert result.stopped_at == 5
    assert result.run_directory == str(base / metadata.UniqueName)
    job = json.loads((base / ".jobs" / "paired-arm.json").read_text())
    assert job["metadata"] == to_wire(metadata) and job["policy_change"] == to_wire(change)
    assert job["fixed_horizon"] is True and job["observe_at"] == [0, 2, 5]
    assert job["result_protocol"] == "smp-research-result-v1" and "summary_path" not in job
    assert not (base / metadata.UniqueName / "observations.json").exists()
    assert metadata == original_metadata
    assert observe_persistent(fake_smp, base, metadata, [0, 2, 5], policy_change=change) == result
    with pytest.raises(ProtocolError, match="manifest changed"):
        observe_persistent(fake_smp, base, metadata, [0, 2, 6], policy_change=change)


def test_persistent_adapter_waits_for_graceful_save_then_resumes(fake_smp, tmp_path, monkeypatch):
    stop = threading.Event()
    monkeypatch.setattr("ehk.micro.persistent.cancellation_event", lambda: stop)
    base = tmp_path / "raw"
    metadata=replace(create_scenario_metadata("resume"),NodeCount=3)
    ready = base / "resume" / "ready"

    def cancel_when_ready():
        deadline = time.monotonic() + 5
        while not ready.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()

    canceller = threading.Thread(target=cancel_when_ready)
    canceller.start()
    try:
        with pytest.raises(KeyboardInterrupt, match="cancelled"):
            observe_persistent(fake_smp, base, metadata, [0, 2])
    finally:
        canceller.join()
    assert (base / "resume" / "resumable").read_text() == "saved"
    assert not (base / "resume" / "observations.json").exists()
    stop.clear()
    assert observe_persistent(fake_smp, base, metadata, [0, 2]).stopped_at == 2


@pytest.mark.parametrize("steps", [[], [-1, 2], [0, 2, 1], [0, 1, 1], [0, True], [0, 1.5]])
def test_persistent_adapter_rejects_invalid_steps_before_launch(fake_smp, tmp_path, steps):
    with pytest.raises(ValueError, match="observation steps"):
        observe_persistent(fake_smp, tmp_path / "raw", create_scenario_metadata("invalid"), steps)
    assert not (tmp_path / "raw").exists()


def test_persistent_window_keeps_absolute_steps_and_exact_trace(fake_smp, tmp_path):
    base, trace = tmp_path / "raw", tmp_path / "exposure" / "late.jsonl.gz"
    result = observe_persistent(
        fake_smp, base, replace(create_scenario_metadata("late"),NodeCount=3), [16, 18, 22], exposure_trace=trace
    )
    assert [point.relative_step for point in result.endpoints] == [0, 2, 6]
    assert result.endpoints[0].exposure.updates == 0
    job = json.loads((base / ".jobs" / "late.json").read_text())
    assert job["trace_start"] == 16 and job["exposure_trace"] == str(trace)
    assert result.exposure_trace.updates == 18 and result.exposure_trace.slots == 36


def test_persistent_adapter_requires_complete_raw_record(fake_smp, tmp_path, monkeypatch):
    import ehk.micro.persistent as persistent

    original = persistent.json.loads

    def invalid_summary(text):
        value = original(text)
        if "result" in value:
            value["result"]["stopped_at"] = -1
        return value

    monkeypatch.setattr(persistent.json, "loads", invalid_summary)
    with pytest.raises(ValueError, match="incomplete persistent observation"):
        observe_persistent(fake_smp, tmp_path / "raw", replace(create_scenario_metadata("incomplete"),NodeCount=3), [0, 2])
