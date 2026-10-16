"""Content-addressed derived artifacts; producers own their scientific dependencies."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from common.io.json import atomic_json, fingerprint, sha


class ArtifactCache:
    def __init__(self, root, *, force=False):
        self.root, self.force = Path(root), force
        self.hits = self.computed = 0

    def materialize(self, name, dependencies, producer):
        key = fingerprint(dependencies)
        target = self.root / name / key
        marker = target / "artifact.json"
        if not self.force and marker.is_file():
            try:
                saved = json.loads(marker.read_text())
                valid = saved.get("key") == key and all(
                    (target / f).resolve().is_relative_to(target.resolve())
                    and (target / f).is_file()
                    and sha(target / f) == h
                    for f, h in saved["files"].items()
                )
            except (OSError, ValueError, KeyError, TypeError):
                valid = False
            if valid:
                self.hits += 1
                return target
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".artifact-", dir=target.parent))
        try:
            producer(stage)
            atomic_json(
                stage / "artifact.json",
                {
                    "key": key,
                    "dependencies": dependencies,
                    "files": {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob("*")) if p.is_file()},
                },
            )
            if target.exists():
                shutil.rmtree(target)
            os.replace(stage, target)
            self.computed += 1
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return target
