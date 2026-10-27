"""Immutable archival of previous attempt receipts, independent of study layout."""

import json

from common.io.json import sha

from .plans import freeze_manifest


def archive_receipt(path, directory):
    if path.exists():
        freeze_manifest(directory / ("receipt-" + sha(path) + ".json"), json.loads(path.read_text()))
