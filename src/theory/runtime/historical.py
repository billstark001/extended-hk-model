"""Read-only adapters for frozen audit jobs and gzip records; no generation/execution."""

import json
from pathlib import Path

from common.experiments.plans import read_plan, unique_tasks
from common.io.json import fingerprint, read_result
from common.runtime.failures import ProtocolError


def metadata_key(meta, exclude=()):
    return fingerprint({k: v for k, v in meta.items() if k not in set(exclude) | {"smoke"}})


class FrozenAuditDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.plan = read_plan(self.root / "plan.json")
        self.binding = json.loads((self.root / "binding.json").read_text())
        if self.binding["plan_fingerprint"] != self.plan["plan_fingerprint"]:
            raise ProtocolError("plan/binding mismatch")
        self.expected = unique_tasks(self.plan["jobs"])

    def path(self, job):
        return self.root / job["family"] / "raw" / (job["id"] + ".json.gz")

    def validate(self, record, job=None):
        saved = record["job"]
        if saved != self.expected.get(saved["id"]) or (job is not None and saved != job):
            raise ProtocolError("raw job differs from the frozen plan")
        source = self.binding["source_fingerprint"]
        if record["build_fingerprint"] != source or record["identity"] != fingerprint({"task": saved, "build": source}):
            raise ProtocolError("raw result/build binding mismatch")
        return record

    def read(self, job):
        return self.validate(read_result(self.path(job)), job)

    def records(self, family):
        records = []
        seen = set()
        for path in sorted((self.root / family / "raw").glob("*.json.gz")):
            if path.name.startswith("."):
                continue
            record = self.validate(read_result(path))
            job = record["job"]
            if job["id"] in seen or job["family"] != family or path != self.path(job):
                raise ProtocolError("raw job differs from the frozen plan")
            seen.add(job["id"])
            records.append(record)
        return records
