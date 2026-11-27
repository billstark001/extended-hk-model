"""Typed document transactions retain arrays across concurrent writes and backup."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
import pytest
from common.experiments.plans import freeze_manifest
from common.storage.documents import DocumentStore, DocumentWriter, DocumentWrite

@dataclass(frozen=True)
class Measurement:
    step: int
    values: np.ndarray

@dataclass(frozen=True)
class Completion:
    complete: bool

@dataclass(frozen=True)
class Configuration:
    revision: int


def test_one_writer_commits_related_typed_values_from_concurrent_workers(tmp_path):
    path=tmp_path/'analysis.sqlite'
    with DocumentWriter(path) as writer:
        def save(step):
            writer.put_many((DocumentWrite('values',str(step),Measurement(step,np.array([step],dtype=np.float32))),
                             DocumentWrite('complete',str(step),Completion(True))))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(save,range(16)))
        assert len(writer.values('values'))==16
    with DocumentStore(path,readonly=True) as reader:
        for step in range(16):
            value=reader.get('values',str(step))
            assert value['step']==step and value['values'].dtype==np.float32
            np.testing.assert_array_equal(value['values'],[step])
            assert reader.get('complete',str(step))=={'complete':True}


def test_failed_related_write_does_not_publish_a_completion_marker(tmp_path):
    with DocumentWriter(tmp_path/'analysis.sqlite') as writer:
        with pytest.raises(ValueError):
            writer.put_many((DocumentWrite('complete','bad',Completion(True)),
                             DocumentWrite('values','bad',Measurement(0,np.array([object()])))))
        assert writer.get('complete','bad') is None
        assert writer.get('values','bad') is None


def test_sqlite_backup_contains_committed_wal_values_and_closes_handles(tmp_path):
    destination=tmp_path/'snapshot.sqlite'
    with DocumentStore(tmp_path/'analysis.sqlite') as writer:
        writer.put('values','one',Measurement(7,np.arange(5,dtype=np.int32)))
        writer.backup(destination)
    with DocumentStore(destination,readonly=True) as reader:
        row=reader.get('values','one')
        assert row['step']==7
        np.testing.assert_array_equal(row['values'],np.arange(5,dtype=np.int32))
    destination.rename(tmp_path/'moved.sqlite')


def test_frozen_manifest_compares_the_serialized_dataclass_contract(tmp_path):
    from common.runtime.failures import ProtocolError
    path=tmp_path/'configuration.json'
    freeze_manifest(path,Configuration(1))
    freeze_manifest(path,Configuration(1))
    with pytest.raises(ProtocolError,match='changed'):
        freeze_manifest(path,Configuration(2))
