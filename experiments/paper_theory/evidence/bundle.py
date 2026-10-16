"""Portable CSV observations: raw readers are confined to export_raw().

Float cells use Python's round-trip representation. Matrix rows consisting only
of zeros are omitted and restored from explicit shapes; no threshold is applied.
Seeds remain decimal/hex strings in tasks.csv and integers in the frozen plan.
"""

import csv
import gzip
import io
import json
import os
import shutil
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from common.experiments.plans import freeze_plan, read_plan
from common.io.json import atomic_json, fingerprint, sha
from common.runtime.failures import ProtocolError

SCHEMA = 1


@contextmanager
def csv_stream(path, mode):
    if str(path).endswith(".gz"):
        with gzip.GzipFile(filename=str(path), mode=mode[0] + "b", compresslevel=3, mtime=0) as binary:
            with io.TextIOWrapper(binary, newline="") as handle:
                yield handle
    else:
        with open(path, mode, newline="") as handle:
            yield handle


def csv_rows(path):
    with csv_stream(path, "rt") as handle:
        yield from csv.DictReader(handle)


def write_rows(path, keys, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with csv_stream(path, "wt") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        writer.writerows(rows)


def number(value):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def write_matrices(directory, arrays, bins):
    """One ordinary CSV column per grid coordinate, explicit row and native shape."""
    info = {}
    for name, value in arrays.items():
        value = np.asarray(value, dtype=float)
        path = directory / (name + ".csv.gz")
        flat = value.ravel()

        def rows():
            for start in range(0, len(flat), bins):
                row = flat[start : start + bins]
                if np.any(row != 0):
                    yield [start // bins, *row.tolist(), *([""] * (bins - len(row)))]

        write_rows(path, ["row", *[f"v{i}" for i in range(bins)]], rows())
        info[name] = {"file": str(path), "shape": list(value.shape), "width": bins}
    return info


def write_frames(directory, frames, alpha, bins):
    keys = sorted({k for f in frames for k, v in f.items() if np.asarray(v).ndim == 0})
    path = directory / "series.csv.gz"
    write_rows(
        path,
        ["frame", "tau", *keys],
        ([i, f["time"] * alpha, *[f.get(k, "") for k in keys]] for i, f in enumerate(frames)),
    )
    arrays: dict[str, dict[int, Any]] = defaultdict(dict)
    for i, frame in enumerate(frames):
        for name, value in frame.items():
            value = np.asarray(value)
            if value.ndim and value.size:
                arrays[name][i] = value
    fields = {}
    for name, values in arrays.items():
        path = directory / (name + ".csv.gz")

        def rows():
            for frame, value in values.items():
                flat = value.ravel()
                for start in range(0, len(flat), bins):
                    row = flat[start : start + bins]
                    if np.any(row != 0):
                        yield [frame, start // bins, *row.tolist(), *([""] * (bins - len(row)))]

        write_rows(path, ["frame", "row", *[f"v{i}" for i in range(bins)]], rows())
        fields[name] = {"file": str(path), "width": bins, "shapes": {str(i): list(v.shape) for i, v in values.items()}}
    return {"series": str(directory / "series.csv.gz"), "arrays": fields}


def export_raw(source, destination, *, force=False):
    """Authenticate native/legacy records once; export every recorded scalar and field."""
    from common.io.records import content_fingerprint, record_path
    from theory.runtime.records import read_micro

    source, destination = Path(source).resolve(), Path(destination).resolve()
    plan = read_plan(source / "plan.json")
    destination.mkdir(parents=True, exist_ok=True)
    freeze_plan(destination / "plan.json", plan)
    stage = Path(tempfile.mkdtemp(prefix=".csv-export-", dir=destination))
    entries, pairs, provenance = {}, {}, []
    previous = None
    if not force and (destination / "data_manifest.json").is_file():
        previous = json.loads((destination / "data_manifest.json").read_text())
        if previous.get("schema") != SCHEMA or previous.get("plan_fingerprint") != plan["plan_fingerprint"]:
            previous = None
    old_hashes = {r["id"]: r["sha256"] for r in previous["provenance"]} if previous else {}
    reused = 0

    def reuse_tree(kind, identifier):
        assert previous is not None
        prefix = f"data/{kind}/{identifier}/"
        for name, expected in previous["files"].items():
            if name.startswith(prefix) and sha(destination / name) != expected:
                raise ProtocolError(f"cached CSV content changed: {name}")
        shutil.copytree(destination / "data" / kind / identifier, stage / kind / identifier, copy_function=os.link)

    initial_hashes = defaultdict(set)
    try:
        for index, task in enumerate(plan["tasks"]):
            identifier, kind, p = task["id"], task["kind"], task["parameters"]
            raw = source / kind / "raw"
            path = raw / (identifier + ".npz") if kind == "measure" else record_path(raw, identifier)
            if not path.is_file():
                continue
            digest = sha(path)
            if previous and identifier in previous["records"] and old_hashes.get(identifier) == digest:
                entry = previous["records"][identifier]
                reuse_tree(kind, identifier)
                if kind == "micro":
                    initial_hashes[(p["population"], task["initial_seed"])].add(entry["initial_hash"])
                    pair_id = entry["pair_id"]
                    if pair_id not in pairs:
                        native_pair = source / "micro" / "pairs" / (pair_id + ".msgpack.lz4")
                        if pair_id in old_hashes:
                            if sha(native_pair) != old_hashes[pair_id]:
                                raise ProtocolError("shared pair content changed")
                            provenance.append(
                                {"id": pair_id, "kind": "pair", "path": str(native_pair), "sha256": old_hashes[pair_id]}
                            )
                        reuse_tree("pair", pair_id)
                        pairs[pair_id] = previous["pairs"][pair_id]
                entries[identifier] = entry
                provenance.append({"id": identifier, "kind": kind, "path": str(path), "sha256": digest})
                reused += 1
                if (index + 1) % 100 == 0:
                    print(f"CSV export {index + 1}/{len(plan['tasks'])}; reused {reused}", flush=True)
                continue
            directory = stage / kind / identifier
            if kind == "measure":
                with np.load(path, allow_pickle=False) as archive:
                    if json.loads(str(archive["metadata"])) != task:
                        raise ProtocolError("raw measure task/plan mismatch")
                    arrays = {k: archive[k] for k in archive.files if k not in ("metadata", "diagnostics")}
                    diagnostic = json.loads(str(archive["diagnostics"]))
                series: dict[int, dict[str, Any]] = defaultdict(dict)
                matrices = {}
                for name, value in arrays.items():
                    if name.startswith(("snapshot_", "final_")) or value.ndim != 1:
                        matrices[name] = value
                    else:
                        series[len(value)][name] = value
                files = []
                for size, columns in series.items():
                    file = directory / f"series_{size}.csv.gz"
                    write_rows(file, list(columns), zip(*columns.values()))
                    files.append(str(file))
                entry = {
                    "kind": kind,
                    "series": files,
                    "arrays": write_matrices(directory, matrices, p["bins"]),
                    "diagnostics": diagnostic,
                }
            else:
                record = read_micro(path)
                if record["task"] != task:
                    raise ProtocolError("raw micro task/plan mismatch")
                result = record["result"]
                initial_hashes[(p["population"], task["initial_seed"])].add(result["initial_hash"])
                pair = result["pair"]
                pair_id = result.get("pair_ref", {}).get("key") or fingerprint(
                    {
                        "parameters": p,
                        "initial_seed": task["initial_seed"],
                        "request_options": task.get("request_options", {}),
                        "initial_hash": result["initial_hash"],
                    }
                )
                pair_hash = content_fingerprint(pair["frames"])
                if pair_id not in pairs:
                    pairs[pair_id] = {
                        **write_frames(stage / "pair" / pair_id, pair["frames"], p["alpha"], p["bins"]),
                        "diagnostics": pair["diagnostics"],
                        "frames_fingerprint": pair_hash,
                    }
                    reference = result.get("pair_ref")
                    if reference:
                        pair_path = path.parent.parent / "pairs" / (pair_id + ".msgpack.lz4")
                        provenance.append(
                            {"id": pair_id, "kind": "pair", "path": str(pair_path), "sha256": reference["sha256"]}
                        )
                elif pairs[pair_id]["frames_fingerprint"] != pair_hash:
                    raise ProtocolError("deterministic shared pair changed between branches")
                entry = {
                    **write_frames(directory, result["frames"], p["alpha"], p["bins"]),
                    "kind": kind,
                    "pair_id": pair_id,
                    "initial_hash": result["initial_hash"],
                    "pair_diagnostics": pair["diagnostics"],
                }
            entries[identifier] = entry
            provenance.append({"id": identifier, "kind": kind, "path": str(path), "sha256": digest})
            if (index + 1) % 50 == 0:
                print(f"CSV export {index + 1}/{len(plan['tasks'])}", flush=True)
        if not entries:
            raise FileNotFoundError("analysis needs at least one declared raw result")
        if any(len(values) != 1 for values in initial_hashes.values()):
            raise ProtocolError("initial state changed under recommendation or process control")

        def relative(value):
            if isinstance(value, dict):
                return {k: relative(v) for k, v in value.items()}
            if isinstance(value, list):
                return [relative(v) for v in value]
            if isinstance(value, str) and value.startswith(str(stage) + "/"):
                return "data/" + str(Path(value).relative_to(stage))
            return value

        task_rows = [
            {
                **t["parameters"],
                "id": t["id"],
                "kind": t["kind"],
                "roles": ";".join(t["roles"]),
                "initial": t.get("initial", ""),
                "branch": t.get("branch", ""),
                "initial_seed": str(t.get("initial_seed", "")),
                "process_seed": str(t.get("process_seed", "")),
                "pair_id": entries.get(t["id"], {}).get("pair_id", ""),
            }
            for t in plan["tasks"]
        ]
        write_rows(stage / "tasks.csv", list(task_rows[0]), ([r[k] for k in task_rows[0]] for r in task_rows))
        write_rows(
            stage / "grids.csv",
            ["bins", "bin", "left", "right", "center"],
            (
                [bins, i, -1 + 2 * i / bins, -1 + 2 * (i + 1) / bins, -1 + 2 * (i + 0.5) / bins]
                for bins in sorted({t["parameters"]["bins"] for t in plan["tasks"]})
                for i in range(bins)
            ),
        )
        write_rows(
            stage / "provenance.csv",
            ["id", "kind", "path", "sha256"],
            ([r[k] for k in ("id", "kind", "path", "sha256")] for r in provenance),
        )
        file_hashes = {"data/" + str(f.relative_to(stage)): sha(f) for f in sorted(stage.rglob("*.csv*"))}
        old = destination / ".previous-data"
        if old.exists():
            shutil.rmtree(old)
        if (destination / "data").exists():
            os.replace(destination / "data", old)
        os.replace(stage, destination / "data")
        atomic_json(
            destination / "data_manifest.json",
            {
                "schema": SCHEMA,
                "plan_fingerprint": plan["plan_fingerprint"],
                "records": relative(entries),
                "pairs": relative(pairs),
                "files": file_hashes,
                "provenance": provenance,
                "bindings": {
                    kind: json.loads((source / kind / "binding.json").read_text())
                    for kind in ("measure", "micro")
                    if (source / kind / "binding.json").is_file()
                },
                "reused_records": reused,
                "float_encoding": "round-trip float64; zero matrix rows restored from native shape",
                "units": {
                    "time": "physical t",
                    "tau": "alpha*t",
                    "rho": "node mass per native bin",
                    "edge": "directed-edge measure mass; source is row, target is column",
                },
                "grid_file": "data/grids.csv",
            },
        )
        if old.exists():
            shutil.rmtree(old)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination


class Bundle:
    """Read and lazily authenticate local CSV files, without consulting raw paths."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.plan = read_plan(self.root / "plan.json")
        self.manifest = json.loads((self.root / "data_manifest.json").read_text())
        if self.manifest["schema"] != SCHEMA or self.manifest["plan_fingerprint"] != self.plan["plan_fingerprint"]:
            raise ProtocolError("CSV bundle schema/plan mismatch")
        self.tasks = {t["id"]: t for t in self.plan["tasks"]}
        self.used = {}

    def path(self, relative):
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root):
            raise ProtocolError("CSV file escapes analysis bundle")
        if relative not in self.used:
            actual = sha(path)
            if actual != self.manifest["files"][relative]:
                raise ProtocolError(f"CSV content changed: {relative}")
            self.used[relative] = actual
        return path

    def matrices(self, info, names=None):
        result = {}
        for name, item in info.items():
            if names is not None and name not in names:
                continue
            value = np.zeros(item["shape"], dtype=float)
            flat, width = value.ravel(), item["width"]
            for row in csv_rows(self.path(item["file"])):
                start = int(row["row"]) * width
                cells = [float(row[f"v{i}"]) for i in range(min(width, len(flat) - start))]
                flat[start : start + len(cells)] = cells
            result[name] = value
        return result

    def frames(self, info, fields=True):
        frames = []
        for row in csv_rows(self.path(info["series"])):
            frames.append({k: number(v) for k, v in row.items() if k not in ("frame", "tau") and v != ""})
        if fields:
            for name, item in info["arrays"].items():
                for index, shape in item["shapes"].items():
                    frames[int(index)][name] = np.zeros(shape, dtype=float)
                width = item["width"]
                for row in csv_rows(self.path(item["file"])):
                    flat = frames[int(row["frame"])][name].ravel()
                    start = int(row["row"]) * width
                    cells = [float(row[f"v{i}"]) for i in range(min(width, len(flat) - start))]
                    flat[start : start + len(cells)] = cells
        return frames

    def dependencies(self, tasks, *, fields=True):
        files = set()
        metadata = {}
        for task in tasks:
            entry = self.manifest["records"][task["id"]]
            items = {task["id"]: entry}
            if task["kind"] == "micro":
                items[entry["pair_id"]] = self.manifest["pairs"][entry["pair_id"]]
            for identifier, item in items.items():
                series = item["series"]
                files.update(series if isinstance(series, list) else [series])
                if fields:
                    files.update(value["file"] for value in item["arrays"].values())
                metadata[identifier] = fingerprint({k: v for k, v in item.items() if fields or k != "arrays"})
        return {
            "files": {name: self.manifest["files"][name] for name in sorted(files) if self.path(name)},
            "metadata": metadata,
        }

    def case(self, task, fields=True):
        entry, p = self.manifest["records"][task["id"]], task["parameters"]
        result = {
            **p,
            "r": p["ratio"],
            "source": task["id"] + ".npz",
            "diagnostics": entry["diagnostics"],
            "converged": entry["diagnostics"]["converged"],
        }
        for file in entry["series"]:
            rows = list(csv_rows(self.path(file)))
            result.update({k: np.array([number(r[k]) for r in rows]) for k in rows[0]})
        names = None if fields is True else (set() if fields is False else fields)
        result.update(self.matrices(entry["arrays"], names))
        return result

    def record(self, task, *, pair_cache=None, fields=True):
        entry = self.manifest["records"][task["id"]]
        pair_id = entry["pair_id"]
        cached = pair_cache if pair_cache is not None else {}
        if pair_id not in cached:
            cached[pair_id] = self.frames(self.manifest["pairs"][pair_id], fields)
        return {
            "task": task,
            "result": {
                "initial_hash": entry["initial_hash"],
                "frames": self.frames(entry, fields),
                "pair": {"frames": cached[pair_id], "diagnostics": entry["pair_diagnostics"]},
                "pair_frames_fingerprint": self.manifest["pairs"][pair_id]["frames_fingerprint"],
            },
        }
