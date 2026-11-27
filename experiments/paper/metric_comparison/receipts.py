"""Decode committed numerical experiment receipts at the SQLite boundary."""
from pathlib import Path
from typing import TypeVar
from common.io.json import fingerprint
from common.storage.documents import DocumentStore
from ehk.domain import parse_dataclass

T = TypeVar("T")


def receipt(directory, namespace, identity, contract: type[T]) -> T | None:
    path = Path(directory) / "analysis.sqlite"
    if not path.is_file():
        return None
    with DocumentStore(path, readonly=True) as store:
        document = store.get(namespace, fingerprint(identity))
    return None if document is None else parse_dataclass(contract, document)
