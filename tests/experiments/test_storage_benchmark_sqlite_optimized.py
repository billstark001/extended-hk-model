"""Correctness checks for the isolated optimized variant; no timed workloads."""

import sqlite3

import numpy as np
import pytest

from experiments.storage_benchmark.backends import _matches, open_backend
from experiments.storage_benchmark.common import document_digest


def document(index=0):
    return {
        "_id": f"record-{index}",
        "origin": "fixture",
        "mode": "gradation",
        "analysis_id": "revision-1",
        "recsys_type": "OpinionM9",
        "decay": 0.05 + index / 100,
        "rewiring": 0.1,
        "repost_rate": 0.2,
        "tolerance": 0.3,
        "post_retain_count": 4,
        "complete": True,
        "stale": False,
        "metadata": {"HKParams": {"Influence": 0.9 - index / 100}, "generation": 0},
        "protocol": {"version": 1},
        "source": {"seed": "18446744073709551615"},
        "payload": {
            "nested": {"values": np.array([[index, index + 1]], dtype=np.float32), "missing": None},
            "counts": [np.array([1, 2], dtype=np.int32), "label", np.float64(0.5)],
            "empty": {},
        },
    }


def test_roundtrip_replacement_schema_projection_and_readonly(tmp_path):
    with open_backend("sqlite_optimized", tmp_path) as backend:
        assert backend.pragmas()["journal_mode"] == "wal"
        assert backend.pragmas()["synchronous"] == 2
        for index in range(3):
            backend.put(document(index))
        assert backend.count() == 3
        for index in range(3):
            assert document_digest(backend.get(f"record-{index}")) == document_digest(document(index))
        record_id = backend.connection.execute("SELECT id FROM records WHERE _id='record-0'").fetchone()[0]
        replacement = document()
        replacement["payload"] = {"replacement": np.array([8, 9], dtype=np.int64)}
        backend.put(replacement)
        assert backend.count() == 3
        assert backend.connection.execute("SELECT id FROM records WHERE _id='record-0'").fetchone()[0] == record_id
        assert document_digest(backend.get("record-0")) == document_digest(replacement)
        assert (
            backend.connection.execute("SELECT COUNT(*) FROM payloads WHERE record_id=?", (record_id,)).fetchone()[0]
            == 1
        )
        columns = {row[1]: row[2] for row in backend.connection.execute("PRAGMA table_info(records)")}
        assert columns["id"] == "INTEGER"
        assert columns["tolerance"] == "REAL" and columns["post_retain_count"] == "INTEGER"
        payload_sql = backend.connection.execute("SELECT sql FROM sqlite_master WHERE name='payloads'").fetchone()[0]
        assert "WITHOUT ROWID" in payload_sql
        assert backend.connection.execute("PRAGMA foreign_key_check").fetchall() == []
        backend.counters.update(payload_array_decodes=0, payload_blob_reads=0)
        statements = []
        backend.connection.set_trace_callback(statements.append)
        assert "payload" not in backend.get("record-0", include_payload=False)
        assert len(backend.query({"origin": "fixture"}, projection=True)) == 3
        assert backend.counters == {"payload_array_decodes": 0, "payload_blob_reads": 0}
        assert not any("payload_layout" in sql or "FROM payloads" in sql for sql in statements)
        backend.connection.set_trace_callback(None)
        assert backend.get("absent") is None
        assert backend.integrity_check() == ["ok"]
        assert backend.checkpoint()["busy"] == 0
    with open_backend("sqlite_optimized", tmp_path, readonly=True) as reader:
        assert document_digest(reader.get("record-0")) == document_digest(replacement)
        with pytest.raises(RuntimeError, match="read-only"):
            reader.put(document())
        with pytest.raises(RuntimeError, match="read-only"):
            reader.update_state("record-0", {"complete": False})
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.connection.execute("DELETE FROM records")


@pytest.mark.parametrize(
    "filters",
    [
        {"origin": "fixture", "decay": {"$gte": 0.06, "$lt": 0.08}},
        {"complete": True, "stale": False},
        {"post_retain_count": {"$gt": 3, "$lte": 4}, "tolerance": {"$in": [0.3]}},
        {"metadata.HKParams.Influence": {"$gte": 0.88, "$lte": 0.9}},
        {"tolerance": None},
        {"tolerance": {"$in": [None, 0.3]}},
        {"metadata.HKParams.Influence": {"$in": [None, 0.9]}},
        {"metadata.extra.value": None},
        {"metadata.extra.value": {"$in": [None, 7]}},
        {"_id": {"$in": [None, "record-0", "record-2"]}},
        {"unknown": {"$in": []}},
        {"hk_influence": None, "record_id": None, "id": None},
        {"hk_influence": 0.9},
    ],
)
def test_filters_match_document_semantics_including_missing_and_null(tmp_path, filters):
    items = [document(index) for index in range(3)]
    items[1]["tolerance"] = None
    del items[2]["tolerance"]
    items[1]["metadata"]["HKParams"]["Influence"] = None
    del items[2]["metadata"]["HKParams"]["Influence"]
    items[0]["metadata"]["extra"] = {"value": 7}
    with open_backend("sqlite_optimized", tmp_path) as backend:
        for item in items:
            backend.put(item)
        expected = {item["_id"] for item in items if _matches(item, filters)}
        assert {item["_id"] for item in backend.query(filters, projection=True)} == expected
        assert {item["_id"] for item in backend.query(filters)} == expected


def test_influence_is_independent_and_internal_column_names_are_not_public_aliases(tmp_path):
    first, second = document(), document(1)
    first.update(hk_influence=0.4, id=123, record_id=456)
    second["decay"] = first["metadata"]["HKParams"]["Influence"]
    second["metadata"]["HKParams"]["Influence"] = first["decay"]
    with open_backend("sqlite_optimized", tmp_path) as backend:
        backend.put(first)
        backend.put(second)
        assert [item["_id"] for item in backend.query({"metadata.HKParams.Influence": 0.05}, projection=True)] == [
            "record-1"
        ]
        assert [item["_id"] for item in backend.query({"decay": 0.05}, projection=True)] == ["record-0"]
        assert [
            item["_id"] for item in backend.query({"hk_influence": 0.4, "id": 123, "record_id": 456}, projection=True)
        ] == ["record-0"]
        backend.update_state("record-0", {"tolerance": 0.6, "post_retain_count": 8})
        assert [item["_id"] for item in backend.query({"tolerance": 0.6, "post_retain_count": 8}, projection=True)] == [
            "record-0"
        ]
        assert backend.get("record-0")["metadata"]["HKParams"]["Influence"] == 0.9


@pytest.mark.parametrize("existing", [False, True])
def test_failed_blob_write_rolls_back_whole_document(tmp_path, existing):
    with open_backend("sqlite_optimized", tmp_path) as backend:
        if existing:
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
        assert not backend.connection.in_transaction
        if existing:
            assert document_digest(backend.get("record-0")) == document_digest(document())
        else:
            assert backend.get("record-0") is None
            assert backend.connection.execute("SELECT COUNT(*) FROM payloads").fetchone()[0] == 0
        assert backend.integrity_check() == ["ok"]
        assert backend.connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_state_write_is_committed_atomically_without_decoding_blobs(tmp_path):
    with open_backend("sqlite_optimized", tmp_path) as backend:
        commits = []
        backend.recovery_before_commit = lambda: commits.append(backend.connection.in_transaction)
        backend.put(document())
        backend.put(document())
        backend.update_state("record-0", {"complete": False, "stale": True, "generation": 7})
        assert commits == [True, True, True]
        assert backend.counters == {"payload_array_decodes": 0, "payload_blob_reads": 0}
        expected = document()
        expected.update(complete=False, stale=True, generation=7)
        assert document_digest(backend.get("record-0")) == document_digest(expected)
        with open_backend("sqlite_optimized", tmp_path, readonly=True) as reader:
            assert document_digest(reader.get("record-0")) == document_digest(expected)
        backend.recovery_before_commit = None
        with pytest.raises(KeyError):
            backend.update_state("absent", {"complete": True})
        assert not backend.connection.in_transaction
        with pytest.raises(ValueError, match="object arrays"):
            backend.put({**document(4), "payload": {"bad": np.array([object()], dtype=object)}})


@pytest.mark.parametrize("query", [False, True])
def test_full_read_holds_one_snapshot_across_concurrent_replacement(tmp_path, monkeypatch, query):
    with open_backend("sqlite_optimized", tmp_path) as writer:
        for index in range(2):
            writer.put(document(index))
        with open_backend("sqlite_optimized", tmp_path, readonly=True) as reader:
            load = reader._load_payload
            injected = False

            def replace_between_header_and_payload(record_id, layout):
                nonlocal injected
                if not injected:
                    injected = True
                    for index in range(2):
                        changed = document(index)
                        changed["metadata"]["generation"] = 5
                        changed["payload"] = {"new": np.array([5.0, 6.0])}
                        writer.put(changed)
                return load(record_id, layout)

            monkeypatch.setattr(reader, "_load_payload", replace_between_header_and_payload)
            observed = reader.query({}) if query else [reader.get("record-0")]
            assert {document_digest(item) for item in observed} == {
                document_digest(document(index)) for index in range(2 if query else 1)
            }
            assert not reader.connection.in_transaction
            monkeypatch.setattr(reader, "_load_payload", load)
            current = reader.query({})
            assert all(item["metadata"]["generation"] == 5 and set(item["payload"]) == {"new"} for item in current)


def test_scope_policy_and_nested_queries_use_composite_indexes(tmp_path):
    with open_backend("sqlite_optimized", tmp_path) as backend:
        backend.put(document())
        scope = {field: document()[field] for field in ("origin", "mode", "analysis_id")}
        policy = {
            **scope,
            **{
                field: document()[field]
                for field in ("recsys_type", "decay", "rewiring", "repost_rate", "post_retain_count", "tolerance")
            },
        }
        for filters, expected_indexes in (
            (scope, ("idx_scope", "idx_policy", "idx_influence")),
            (policy, ("idx_policy",)),
            ({**scope, "metadata.HKParams.Influence": 0.9}, ("idx_influence",)),
        ):
            where, parameters = backend._compile_filters(filters)
            detail = " ".join(
                row[3]
                for row in backend.connection.execute(
                    "EXPLAIN QUERY PLAN SELECT document_json FROM records" + where, parameters
                )
            )
            # SQLite may choose any index with the same complete scope prefix.
            assert any("SEARCH records USING INDEX " + name in detail for name in expected_indexes)
            assert "origin=? AND mode=? AND analysis_id=?" in detail
            assert [row["_id"] for row in backend.query(filters, projection=True)] == ["record-0"]
