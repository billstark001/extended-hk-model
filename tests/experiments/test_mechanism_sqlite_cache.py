"""Mechanism inputs are typed SQLite values bound to their actual raw source."""

from dataclasses import dataclass
from pathlib import Path
from ehk.domain import ScenarioMetadata
from experiments.paper.analysis.models import AnalysisProtocol
from ehk.micro.scenarios import create_scenario_metadata

import numpy as np
import numpy.testing

from common.storage.documents import DocumentStore
from experiments.paper.mechanism_cases import plot_supplement as mechanism


def test_mechanism_cache_reuses_arrays_and_invalidates_same_name_changed_source(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    metadata = raw / "metadata.json"
    metadata.write_text("{}")
    calls = []
    @dataclass
    class Collector:
        active_step: int = 2
        def set_state(self,**kwargs):
            calls.append(1)
    collector = Collector()
    monkeypatch.setattr(mechanism, "c", collector)
    monkeypatch.setattr(mechanism, "CACHE_FILE_PATH", tmp_path / "analysis.sqlite")
    monkeypatch.setattr(mechanism, "analysis_protocol", lambda: AnalysisProtocol(metric_revision=6))
    @dataclass
    class SavedRecord:
        unique_name: str
        directory: Path
        metadata: ScenarioMetadata
        max_step: int
        opinions: np.ndarray
        agent_numbers: np.ndarray
        agent_opinion_sums: np.ndarray
    record = SavedRecord(unique_name="same-name", directory=raw, metadata=create_scenario_metadata("same-name"), max_step=2,
        opinions=np.array([[-.5, .5], [-.4, .4], [-.3, .3]], dtype=np.float32),
        agent_numbers=np.ones((3, 2, 2), dtype=np.int64), agent_opinion_sums=np.ones((3, 2, 2)))
    first = mechanism.get_basic_data(record)
    cached = mechanism.get_basic_data(record)
    assert len(calls) == 1
    np.testing.assert_array_equal(first.x_seq, cached.x_seq)
    assert cached.x_seq.dtype == np.float32
    metadata.write_text('{"changed": true}')
    mechanism.get_basic_data(record)
    assert len(calls) == 2
    with DocumentStore(tmp_path / "analysis.sqlite", readonly=True) as store:
        assert len(list(store.values("mechanism.basic"))) == 2
    assert not list(tmp_path.glob("*.pkl"))
