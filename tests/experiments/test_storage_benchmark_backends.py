"""Small correctness checks for isolated adapters; these are not benchmarks."""

import sqlite3
from importlib.util import find_spec

import numpy as np
import pytest

from experiments.storage_benchmark.backends import open_backend

BACKENDS = ["current", "indexed_split"] + (["tinymongo"] if find_spec("tinymongo") else [])


def document(index=0):
    return {
        "_id": f"doc-{index}",
        "origin": "fixture",
        "mode": "gradation",
        "analysis_id": "rev-1",
        "recsys_type": "OpinionM9",
        "decay": 0.05 + index / 100,
        "rewiring": 0.05,
        "repost_rate": 0.1,
        "complete": True,
        "stale": False,
        "metadata": {"HKParams": {"Influence": 0.05}, "generation": index, "seed": "18446744073709551615"},
        "protocol": {"revision": 3},
        "source": {"status": "fixture"},
        "payload": {"p_index": np.array([0.1, 0.2], dtype=np.float64), "x_indices": np.array([0, 1], dtype=np.int32)},
    }


@pytest.mark.parametrize("name", BACKENDS)
def test_roundtrip_queries_state_and_readonly(name, tmp_path):
    directory = tmp_path / name
    with open_backend(name, directory) as backend:
        assert backend.pragmas()["journal_mode"] == "wal"
        assert backend.pragmas()["synchronous"] == 2
        for i in range(3):
            backend.put(document(i))
        backend.put(document(0))
        assert backend.count() == 3
        found = backend.get("doc-0")
        assert found["metadata"]["seed"] == "18446744073709551615"
        for field, expected in document()["payload"].items():
            assert found["payload"][field].dtype == expected.dtype
            np.testing.assert_array_equal(found["payload"][field], expected)
        filters = {"origin": "fixture", "decay": {"$gte": 0.06, "$lt": 0.08}, "complete": True}
        projected = backend.query(filters, projection=True)
        assert {row["_id"] for row in projected} == {"doc-1", "doc-2"}
        assert all("payload" not in row for row in projected)
        assert "payload" not in backend.get("doc-0", include_payload=False)
        nested = backend.query({"metadata.HKParams.Influence": 0.05}, projection=True)
        assert len(nested) == 3
        backend.update_state("doc-0", {"complete": False, "stale": True, "generation": 8})
        changed = backend.get("doc-0")
        assert not changed["complete"] and changed["stale"] and changed["generation"] == 8
        np.testing.assert_array_equal(changed["payload"]["p_index"], document()["payload"]["p_index"])
        assert backend.integrity_check() == ["ok"]
        assert backend.checkpoint()["busy"] == 0
    with open_backend(name, directory, readonly=True) as reader:
        assert reader.count() == 3 and reader.get("doc-0")["generation"] == 8
        with pytest.raises(RuntimeError, match="read-only"):
            reader.put(document(9))
        with pytest.raises(RuntimeError, match="read-only"):
            reader.update_state("doc-0", {"complete": True})


@pytest.mark.parametrize("name", BACKENDS)
def test_projection_decode_counters_and_recovery_hook(name, tmp_path):
    with open_backend(name, tmp_path / name) as backend:
        backend.put(document())
        backend.counters["payload_array_decodes"] = 0
        backend.counters["payload_blob_reads"] = 0
        assert len(backend.query({"origin": "fixture"}, projection=True)) == 1
        if name == "current":
            assert backend.counters["payload_array_decodes"] == 2
        else:
            assert backend.counters["payload_array_decodes"] == 0
        if name == "indexed_split":
            assert backend.counters["payload_blob_reads"] == 0
        commits = []
        backend.recovery_before_commit = lambda: commits.append("before-commit")
        backend.put(document(1))
        backend.update_state("doc-0", {"generation": 2})
        assert commits == ["before-commit", "before-commit"]
        backend.recovery_before_commit = None
        backend.put(document(2))
        assert len(commits) == 2


def test_split_payload_replacement_and_metadata_snapshot(tmp_path):
    with open_backend("indexed_split", tmp_path) as backend:
        backend.put(document())
        revised = document()
        revised["payload"] = {"other": np.array([3], dtype=np.int64)}
        revised["metadata"]["generation"] = 3
        backend.put(revised)
        found = backend.get("doc-0")
        assert set(found["payload"]) == {"other"} and found["metadata"]["generation"] == 3
        assert backend.query({"unknown": {"$in": [None]}}, projection=True)
        with pytest.raises(ValueError, match="object arrays"):
            backend.put({**document(5), "payload": {"bad": np.array([object()], dtype=object)}})


def test_split_failed_payload_insert_rolls_back_metadata_and_all_payloads(tmp_path):
    with open_backend("indexed_split", tmp_path) as backend:
        backend.put(document())
        backend.connection.execute(
            "CREATE TEMP TRIGGER fail_payload BEFORE INSERT ON payloads WHEN NEW.name='[\"fail\"]' "
            "BEGIN SELECT RAISE(ABORT,'fixture payload failure'); END"
        )
        changed = document()
        changed["metadata"]["generation"] = 9
        changed["payload"] = {"fail": np.array([9])}
        with pytest.raises(sqlite3.IntegrityError, match="fixture payload failure"):
            backend.put(changed)
        restored = backend.get("doc-0")
        assert restored["metadata"]["generation"] == 0
        assert set(restored["payload"]) == {"p_index", "x_indices"}
        assert backend.integrity_check() == ["ok"]


def test_split_full_read_uses_one_snapshot_across_header_and_payload(tmp_path, monkeypatch):
    with open_backend("indexed_split", tmp_path) as writer:
        writer.put(document())
        with open_backend("indexed_split", tmp_path, readonly=True) as reader:
            load = reader._load_payload

            def write_between_header_and_blob(identifier):
                changed = document()
                changed["metadata"]["generation"] = 5
                changed["payload"]["p_index"] = np.array([5.0, 6.0])
                writer.put(changed)
                return load(identifier)

            monkeypatch.setattr(reader, "_load_payload", write_between_header_and_blob)
            observed = reader.get("doc-0")
            assert observed["metadata"]["generation"] == 0
            np.testing.assert_array_equal(observed["payload"]["p_index"], [0.1, 0.2])
            monkeypatch.setattr(reader, "_load_payload", load)
            current = reader.get("doc-0")
            assert current["metadata"]["generation"] == 5
            np.testing.assert_array_equal(current["payload"]["p_index"], [5.0, 6.0])


@pytest.mark.parametrize("name", BACKENDS)
def test_recursive_payload_roundtrip_and_projection(name, tmp_path):
    item = document()
    item["payload"] = {
        "reference": {"observed": np.array([[0.2, 0.3]], dtype=np.float32), "missing": None},
        "stages": [{"counts": np.array([1, 2, 3], dtype=np.int64)}, "label", np.float64(0.5)],
        "empty": {},
    }
    with open_backend(name, tmp_path / name) as backend:
        backend.put(item)
        found = backend.get(item["_id"])
        np.testing.assert_array_equal(
            found["payload"]["reference"]["observed"], item["payload"]["reference"]["observed"]
        )
        assert found["payload"]["reference"]["observed"].dtype == np.float32
        assert found["payload"]["reference"]["missing"] is None
        np.testing.assert_array_equal(found["payload"]["stages"][0]["counts"], [1, 2, 3])
        assert found["payload"]["stages"][1:] == ["label", 0.5]
        assert found["payload"]["empty"] == {}
        assert "payload" not in backend.query({"origin": "fixture"}, projection=True)[0]
