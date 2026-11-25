"""Corpus identity and source transformations, independent of database adapters."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from common.io.json import atomic_json
from experiments.storage_benchmark import corpus


def _array_digest(value):
    if isinstance(value, np.ndarray):
        return [str(value.dtype), value.shape, hashlib.sha256(value.tobytes()).hexdigest()]
    if isinstance(value, dict):
        return {key: _array_digest(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_array_digest(item) for item in value]
    return value


@pytest.fixture
def fixture_sources(tmp_path, monkeypatch):
    metadata = {
        "UniqueName": "fixture",
        "HKParams": {"Influence": 0.05, "RewiringRate": 0.1, "RepostRate": 0.25, "Tolerance": 0.45},
        "RecsysFactoryType": "OpinionM9",
        "PostRetainCount": 2,
    }
    tasks = [
        {
            "id": f"s_fixture_{index}",
            "mode": "epsilon"
            if index < 1600
            else "gradation"
            if index < 13888
            else "mech"
            if index < 13899
            else "replicate",
            "metadata": metadata,
        }
        for index in range(corpus.FULL_PAPER_COUNT)
    ]
    path = tmp_path / "actual.json.gz"
    arm = {
        "endpoints": [
            {"step": 0, "pathway": {"separation": 0.2}, "exposure": {"rewires": 0}},
            {"step": 16, "pathway": {"separation": 0.4}, "exposure": {"rewires": 3}},
        ],
        "final_opinions": [-0.5, 0.5],
        "raw_record": "raw/fixture",
        "stopped_at": 16,
        "stop_reason": "horizon",
        "terminal": {},
        "rng": {"Seed1": "0x01"},
    }
    with gzip.open(path, "wt") as stream:
        json.dump({"task_fingerprint": "actual-task", "result": {"reference": arm, "stages": {}}}, stream)
    sources = {
        "tasks": tasks,
        "schema": {
            "protocol": {"verification": "real-schema-template"},
            "source": {"directory": "original", "files": [{"sha256": "real-source"}]},
        },
        "actual_files": [path],
        "actual_tasks": {"actual": {"metadata": metadata}},
        "actual_protocol": {"cadence": 16},
        "provenance": {
            "actual_records": [{"path": "actual.json.gz", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
            "actual_plan": {"sha256": "actual-plan"},
            "observed_reference_lengths": {2: 1},
            "observed_branch_lengths": {},
        },
        "lengths": np.array([33, 129]),
        "length_probabilities": np.array([3824, 496]) / 4320,
        "final_opinion_size": 500,
        # Deliberately nonalphabetical. Both differently shaped grids consume
        # RNG, so JSON key sorting must not reassign their random samples.
        "probe_shapes": {"z_force": [11, 201], "a_force": [3, 7], "x": [201], "potential": [11, 201]},
    }
    monkeypatch.setattr(corpus, "load_sources", lambda: sources)
    return sources


def test_original_scale_and_versioned_identity_are_explicit(fixture_sources):
    assert corpus.get_profile("actual_available")["count"] == 1
    assert corpus.get_profile("full_paper")["count"] == 14099
    versioned = corpus.get_profile("versioned_2x")
    assert versioned["count"] == 28198
    assert versioned["data_kind"] == "research-scale_calibrated_synthetic"
    assert any("not simulations" in item for item in versioned["assumptions"])
    first = corpus.document_at(0, "full_paper")
    assert _array_digest(first) == _array_digest(corpus.document_at(0, "versioned_2x"))
    revised = corpus.document_at(14099, "versioned_2x")
    assert first["_id"] != revised["_id"]
    assert first["name"] == revised["name"]
    assert first["analysis_id"] != revised["analysis_id"]
    assert not np.array_equal(first["payload"]["final_opinions"], revised["payload"]["final_opinions"])


def test_document_regeneration_is_order_independent_and_mutation_safe(fixture_sources):
    before = _array_digest(corpus.document_at(12))
    different = corpus.document_at(13)
    assert before != _array_digest(different)
    document = corpus.document_at(12)
    document["metadata"]["HKParams"]["Influence"] = 999
    document["protocol"]["verification"] = "changed"
    document["payload"]["final_opinions"][:] = 999
    assert _array_digest(corpus.document_at(12)) == before
    assert "synthetic" in corpus.document_at(12)["protocol"]["verification"]


def test_payload_shapes_use_measured_sizes_and_mechanism_grids(fixture_sources):
    document = corpus.document_at(0)
    payload = document["payload"]
    assert payload["final_opinions"].shape == (500,)
    assert payload["p_index"].dtype == np.dtype("float64")
    assert len(payload["x_indices"]) in (33, 129)
    assert np.all(np.diff(payload["x_indices"]) > 0)
    mechanism = corpus.document_at(13888)
    assert mechanism["payload"]["force_probe.potential"].shape == (11, 201)
    assert mechanism["payload"]["force_probe.x"].shape == (201,)
    assert all(isinstance(value, np.ndarray) for value in mechanism["payload"].values())
    assert len(list(corpus.iter_documents(limit=3))) == 3


def test_actual_profile_preserves_observed_values_and_measure_names(fixture_sources):
    document = corpus.document_at(0, "actual_available")
    assert document["source"]["kind"] == "actual_exported_summary"
    assert document["metadata"] == fixture_sources["actual_tasks"]["actual"]["metadata"]
    np.testing.assert_array_equal(document["payload"]["reference.final_opinions"], [-0.5, 0.5])
    observations = document["payload"]
    np.testing.assert_array_equal(observations["reference.step"], [0, 16])
    np.testing.assert_array_equal(observations["reference.pathway.separation"], [0.2, 0.4])
    np.testing.assert_array_equal(observations["reference.exposure.rewires"], [0, 3])
    assert "p_index" not in document["payload"]
    assert document["source"]["raw_records_not_loaded"] == ["raw/fixture"]


def test_actual_source_mutation_is_detected(fixture_sources):
    path = fixture_sources["actual_files"][0]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed after inventory"):
        corpus.document_at(0, "actual_available")


def _source_pack_roundtrip(sources, path):
    """Match prepare's atomic_json and worker's source-pack type restoration."""
    packed = dict(sources)
    packed["lengths"] = sources["lengths"].tolist()
    packed["length_probabilities"] = sources["length_probabilities"].tolist()
    packed["actual_files"] = [str(item) for item in sources["actual_files"]]
    atomic_json(path, packed)
    restored = json.loads(path.read_text())
    restored["lengths"] = np.asarray(restored["lengths"], dtype=np.int64)
    restored["length_probabilities"] = np.asarray(restored["length_probabilities"], dtype=float)
    restored["actual_files"] = [Path(item) for item in restored["actual_files"]]
    return restored


def _payload_signatures(document):
    """Independent signatures for each array, without the benchmark codec."""
    return {
        name: {
            "dtype": array.dtype.str,
            "shape": array.shape,
            "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
        for name, array in document["payload"].items()
    }


def test_sorted_source_pack_preserves_each_array_and_revision(fixture_sources, tmp_path, monkeypatch):
    cases = [
        (0, "full_paper"),
        (1599, "full_paper"),
        (1600, "full_paper"),
        (13887, "full_paper"),
        (13888, "full_paper"),
        (13898, "full_paper"),
        (13899, "full_paper"),
        (14098, "full_paper"),
        (14099, "versioned_2x"),
        (27987, "versioned_2x"),
        (27997, "versioned_2x"),
        (28197, "versioned_2x"),
        (0, "actual_available"),
    ]
    before = {(index, profile): corpus.document_at(index, profile) for index, profile in cases}
    restored = _source_pack_roundtrip(fixture_sources, tmp_path / "sources-pack.json")
    assert list(fixture_sources["probe_shapes"]) != list(restored["probe_shapes"])
    assert list(restored["probe_shapes"]) == sorted(fixture_sources["probe_shapes"])
    assert restored["lengths"].dtype == np.dtype("int64")
    assert restored["length_probabilities"].dtype == np.dtype("float64")
    assert all(isinstance(path, Path) for path in restored["actual_files"])
    monkeypatch.setattr(corpus, "load_sources", lambda: restored)
    for index, profile in cases:
        after = corpus.document_at(index, profile)
        expected = before[index, profile]
        assert _payload_signatures(after) == _payload_signatures(expected), (index, profile)
        assert _array_digest(after) == _array_digest(expected), (index, profile)
    assert _payload_signatures(corpus.document_at(13888, "versioned_2x")) == _payload_signatures(
        before[13888, "full_paper"]
    )
    assert _payload_signatures(before[13888, "full_paper"]) != _payload_signatures(before[27987, "versioned_2x"])


def test_mechanism_generation_ignores_reversed_shape_dictionary(fixture_sources, monkeypatch):
    indices = (13888, 13898, 27987, 27997)
    before = {index: corpus.document_at(index, "versioned_2x") for index in indices}
    reversed_sources = {
        **fixture_sources,
        "probe_shapes": dict(reversed(list(fixture_sources["probe_shapes"].items()))),
    }
    monkeypatch.setattr(corpus, "load_sources", lambda: reversed_sources)
    for index in indices:
        after = corpus.document_at(index, "versioned_2x")
        assert _payload_signatures(after) == _payload_signatures(before[index]), index
        names = [name for name in after["payload"] if name.startswith("force_probe.")]
        assert names == sorted(names)


@pytest.mark.parametrize("index", [-1, 14099, True, 1.5])
def test_invalid_indices_fail_instead_of_wrapping(index, fixture_sources):
    with pytest.raises(IndexError):
        corpus.document_at(index)


def test_invalid_profiles_and_limits_fail(fixture_sources):
    with pytest.raises(ValueError):
        corpus.get_profile("imaginary")
    with pytest.raises(ValueError):
        list(corpus.iter_documents(limit=-1))
