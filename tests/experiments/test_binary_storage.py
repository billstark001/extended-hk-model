"""Synthetic records and injected failures; never execute scientific solvers."""

import errno
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from common.io.json import atomic_json, fingerprint, sha
from common.io.records import equivalent, read_record, write_record
from common.runtime.failures import BatchTaskError, ProtocolError
from common.runtime.process import run_tasks
from common.runtime.storage import RunStorage
from theory.runtime import completed
from theory.runtime.records import read_micro


def test_lossless_binary_arrays_and_footer(tmp_path):
    path = tmp_path / "record.msgpack.lz4"
    values = np.array([0.0, -0.0, np.nextafter(0, 1), 1 / 3], dtype="<f8")
    result = {"frames": [{"time": 0.0, "rho": values.reshape(2, 2)}], "large_uints": np.array([2**63 + 1], dtype="<u8")}
    write_record(path, {"id": "same", "seed": 2**63 + 17}, result)
    read = read_record(path)
    assert read["task"]["seed"] == 2**63 + 17
    assert read["result"]["frames"][0]["rho"].tobytes() == values.tobytes()
    assert read["result"]["large_uints"].tobytes() == result["large_uints"].tobytes()
    path.write_bytes(path.read_bytes()[:-12])
    with pytest.raises((RuntimeError, ValueError, EOFError)):
        read_record(path)


def test_legacy_json_still_reads_without_any_generator(tmp_path):
    path = tmp_path / "old.json"
    original = {"task": {"id": "old"}, "result": {"frames": [{"time": 0, "rho": [0, 1]}]}}
    path.write_text(json.dumps(original))
    assert equivalent(original, read_record(path))


def test_external_write_retry_retains_computed_record(tmp_path, monkeypatch):
    root = tmp_path / "bulk"
    root.mkdir()
    store = RunStorage(root, tmp_path / "local")
    destination = root / "raw" / "same.record"
    writes, replacements = [], []
    original_replace = __import__("os").replace

    def injected(source, target):
        if Path(target) == destination:
            replacements.append(target)
            if len(replacements) == 1:
                raise OSError(errno.EIO, "cable disturbed")
        return original_replace(source, target)

    monkeypatch.setattr("common.runtime.storage.os.replace", injected)
    monkeypatch.setattr("common.runtime.storage.time.sleep", lambda seconds: None)

    def writer(path):
        writes.append(path)
        path.write_bytes(b"already computed")

    store.commit(destination, writer)
    assert len(writes) == 1 and len(replacements) == 2
    assert destination.read_bytes() == b"already computed"
    assert not list(store.spool.glob("*.pending.json"))


def test_spool_recovery_after_interruption_does_not_recompute(tmp_path, monkeypatch):
    root = tmp_path / "bulk"
    root.mkdir()
    store = RunStorage(root, tmp_path / "local")
    marker = store.marker.read_bytes()
    store.marker.unlink()  # Simulate the bulk volume becoming unavailable.
    monkeypatch.setattr("common.runtime.storage.time.sleep", lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt()))
    destination = root / "raw" / "task.record"
    with pytest.raises(KeyboardInterrupt):
        store.commit(destination, lambda path: path.write_bytes(b"retained result"))
    assert list(store.spool.glob("*.pending.json"))
    store.marker.write_bytes(marker)
    restored = RunStorage(root, tmp_path / "local")
    restored.recover()
    assert destination.read_bytes() == b"retained result"
    assert not list(restored.spool.glob("*.pending.json"))


def test_wrong_volume_owner_is_fatal(tmp_path):
    root = tmp_path / "bulk"
    root.mkdir()
    store = RunStorage(root, tmp_path / "local")
    atomic_json(store.marker, {"root": "other run"})
    with pytest.raises(ProtocolError):
        store.read(lambda: pytest.fail("wrong-volume data accessed"))


def test_io_error_is_never_classified_as_truncated_scientific_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(completed, "read_record", lambda path: (_ for _ in ()).throw(OSError(errno.EIO, "offline")))
    with pytest.raises(OSError) as error:
        completed.cached(tmp_path / "fake.msgpack.lz4", {}, "micro")
    assert error.value.errno == errno.EIO


def test_ordinary_failure_isolated_but_global_identity_failure_cancels():
    calls = []

    def isolated(task):
        calls.append(task)
        if task == 1:
            raise RuntimeError("scientific failure; do not retry")

    with pytest.raises(BatchTaskError):
        run_tasks(isolated, range(4), 1, isolate_failures=True)
    assert calls == [0, 1, 2, 3]
    calls.clear()

    def fatal(task):
        calls.append(task)
        raise ProtocolError("source mismatch")

    with pytest.raises(ProtocolError):
        run_tasks(fatal, range(4), 1, isolate_failures=True)
    assert calls == [0]


def test_transport_retries_keep_same_task_and_seed():
    calls = []
    task = {"id": "same", "process_seed": 123}

    def transient(item):
        calls.append(dict(item))
        if len(calls) < 3:
            raise subprocess.CalledProcessError(1, ["synthetic adapter"])

    run_tasks(transient, [task], 1, isolate_failures=True)
    assert calls == [task, task, task]


def test_shared_pair_reference_validates_parameters_and_file_hash(tmp_path):
    raw = tmp_path / "micro" / "raw"
    task = {"id": "branch", "parameters": {"epsilon": 0.4}, "initial_seed": 1}
    identity = {k: v for k, v in task.items() if k != "id"}
    key = fingerprint(identity)
    pair = tmp_path / "micro" / "pairs" / (key + ".msgpack.lz4")
    write_record(pair, {**identity, "id": key}, {"initial_hash": "fixed", "pair": {"frames": []}})
    branch = raw / "branch.msgpack.lz4"
    write_record(branch, task, {"initial_hash": "fixed", "frames": [], "pair_ref": {"key": key, "sha256": sha(pair)}})
    assert read_micro(branch)["result"]["pair"] == {"frames": []}
    pair.write_bytes(pair.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        read_micro(branch)


def test_incomplete_pair_cannot_mask_a_different_cached_task(tmp_path):
    path = tmp_path / "micro" / "raw" / "same.msgpack.lz4"
    original = {"id": "same", "parameters": {"epsilon": 0.4}}
    write_record(path, original, {"frames": [], "pair_ref": {"key": "a" * 64, "sha256": "missing"}})
    with pytest.raises(ProtocolError, match="protocol differs"):
        completed.cached(path, {**original, "parameters": {"epsilon": 0.2}}, "micro")
