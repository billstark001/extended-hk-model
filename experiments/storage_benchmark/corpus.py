"""Read-only research sources and deterministic, explicitly synthetic extensions.

No benchmark timer or database adapter belongs in this module. Numerical arrays
remain NumPy values until a shared benchmark codec prepares them for a backend.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.paper.scenarios import TASKS, execution_tasks

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs/paper/recommendation_manuscript/completed_20261119/data/study"
SCHEMA_DATABASE = ROOT / "outputs/experiments/paper/reanalysis_smoke_20261122/stats-final.sqlite"
PROBE_DIRECTORY = ROOT / "outputs/experiments/paper/mechanism_cases/social_force_probe"
CORPUS_SEED = 2026112201
CORPUS_REVISION = "research-storage-corpus-v2"
FULL_PAPER_COUNT = 14099
PROFILES = ("actual_available", "full_paper", "versioned_2x")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json_bytes(value):
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _source(path, **values):
    return {"path": str(Path(path).relative_to(ROOT)), "sha256": _sha(path), **values}


@lru_cache(maxsize=1)
def load_sources():
    """Measure available exports; fail rather than silently invent missing sources.

    All SQLite access is read-only. This caches small schema/provenance objects,
    not the full corpus or all exported trajectories.
    """
    required = [SCHEMA_DATABASE, STUDY / "plan.json", PROBE_DIRECTORY / "counterfactual_probe_baseline.npz"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"storage corpus source is unavailable: {path}")
    with closing(sqlite3.connect(SCHEMA_DATABASE.resolve().as_uri() + "?mode=ro", uri=True)) as database:
        rows = database.execute("SELECT body FROM documents ORDER BY namespace,key").fetchall()
    if not rows:
        raise ValueError("the real analysis schema database is empty")
    schemas = [json.loads(row[0]) for row in rows]
    schema = next((value for value in schemas if value.get("name", "").endswith("_full")), schemas[0])
    plan = json.loads((STUDY / "plan.json").read_text())
    tasks = {task["id"]: task for task in plan["tasks"]}
    files = sorted((STUDY / "records").glob("*.json.gz"))
    if not files:
        raise ValueError("no actual staged summary records are available")
    references, branches, final_sizes = Counter(), Counter(), Counter()
    real_files = []
    for path in files:
        with gzip.open(path, "rt") as handle:
            envelope = json.load(handle)
        result = envelope["result"]
        if path.stem.removesuffix(".json") not in tasks:
            raise ValueError(f"actual summary is outside its plan: {path}")
        reference = result["reference"]
        references[len(reference["endpoints"])] += 1
        final_sizes[len(reference["final_opinions"])] += 1
        for stage in result.get("stages", {}).values():
            for arm in stage["arms"].values():
                branches[len(arm["endpoints"])] += 1
                final_sizes[len(arm["final_opinions"])] += 1
        real_files.append(_source(path, task_id=path.stem.removesuffix(".json")))
    probe_files = sorted(PROBE_DIRECTORY.glob("counterfactual_probe_*.npz"))
    probe_shapes = {}
    for path in probe_files:
        with np.load(path, allow_pickle=False) as arrays:
            shapes = {
                name: list(arrays[name].shape)
                for name in arrays.files
                if arrays[name].ndim and arrays[name].dtype.kind in "fiub"
            }
        if probe_shapes and shapes != probe_shapes:
            raise ValueError("observed mechanism probe array shapes differ")
        probe_shapes = shapes
    paper_tasks = []
    for mode in TASKS.names:
        for task in execution_tasks(mode):
            paper_tasks.append({**task, "mode": mode})
    if len(paper_tasks) != FULL_PAPER_COUNT:
        raise ValueError("original paper task count changed; revise this corpus explicitly")
    combined = references + branches
    lengths = np.array(sorted(combined), dtype=np.int64)
    counts = np.array([combined[int(length)] for length in lengths], dtype=np.float64)
    schema_measurements = [
        {
            "name": item["name"],
            "field_count": len(item),
            "encoded_document_bytes": len(row[0].encode()),
            "metadata_bytes": _json_bytes(item["metadata"]),
            "protocol_bytes": _json_bytes(item["protocol"]),
            "source_bytes": _json_bytes(item["source"]),
            "row_sha256": hashlib.sha256(row[0].encode()).hexdigest(),
        }
        for row, item in zip(rows, schemas, strict=True)
    ]
    return {
        "schema": schema,
        "tasks": paper_tasks,
        "actual_tasks": tasks,
        "actual_files": files,
        "actual_protocol": plan["protocol"],
        "lengths": lengths,
        "length_probabilities": counts / counts.sum(),
        "final_opinion_size": final_sizes.most_common(1)[0][0],
        "probe_shapes": probe_shapes,
        "provenance": {
            "schema_database": _source(SCHEMA_DATABASE),
            "schema_rows": schema_measurements,
            "actual_plan": _source(STUDY / "plan.json"),
            "actual_records": real_files,
            "mechanism_probes": [_source(path) for path in probe_files],
            "original_scenario_source": _source(ROOT / "experiments/paper/scenarios.py"),
            "original_mode_counts": dict(Counter(task["mode"] for task in paper_tasks)),
            "observed_reference_lengths": dict(references),
            "observed_branch_lengths": dict(branches),
            "observed_final_opinion_sizes": dict(final_sizes),
            "observed_probe_shapes": probe_shapes,
            "unavailable": ["full original-paper wide statistics and bulk raw runs on the unmounted external volume"],
        },
    }


def get_profile(profile="full_paper"):
    """Return count and auditable measured-source/assumption information."""
    if profile not in PROFILES:
        raise ValueError(f"unknown corpus profile: {profile}")
    sources = load_sources()
    actual = profile == "actual_available"
    revisions = 2 if profile == "versioned_2x" else 1
    return {
        "profile": profile,
        "count": len(sources["actual_files"]) if actual else FULL_PAPER_COUNT * revisions,
        "revisions": revisions,
        "corpus_revision": CORPUS_REVISION,
        "seed": CORPUS_SEED,
        "data_kind": "transformed_actual_summaries" if actual else "research-scale_calibrated_synthetic",
        "source_type": "actual_saved_staged_summaries" if actual else "synthetic_extension_of_real_task_coordinates",
        "array_length_distribution": {
            "measured_reference": deepcopy(sources["provenance"]["observed_reference_lengths"]),
            "measured_branches": deepcopy(sources["provenance"]["observed_branch_lengths"]),
            "synthetic_point_lengths": sources["lengths"].tolist() if not actual else None,
            "synthetic_point_probabilities": sources["length_probabilities"].tolist() if not actual else None,
            "final_opinions": sources["final_opinion_size"],
            "mechanism_grids": deepcopy(sources["probe_shapes"]),
        },
        "provenance": deepcopy(sources["provenance"]),
        "assumptions": []
        if actual
        else [
            "The complete original-paper task coordinates are real, but generated numerical results are not simulations.",
            "33/129-point probabilities are measured across available staged branches/reference runs, not the offline original scan.",
            "The measured staged length mixture is transferred to the original-paper analysis schema as a storage-size assumption.",
            "500 final opinions are included as a measured summary shape, not as an observed field of every original scan statistic.",
            "Synthetic horizons are independently chosen between 1024 and the original 20000-step cap; their distribution is not an observed stopping-time estimate.",
            "Only the 11 mechanism tasks receive probe grids, matching the available 11 probe exports and their exact dimensions.",
            "Additional analysis revisions contain independently perturbed synthetic arrays; no dataset-wide constant padding is used.",
        ],
    }


def _scalars(metadata, *, identifier, origin, mode, name, analysis_id):
    parameters = metadata["HKParams"]
    return {
        "_id": identifier,
        "origin": origin,
        "mode": mode,
        "analysis_id": analysis_id,
        "name": name,
        "recsys_type": metadata["RecsysFactoryType"],
        "decay": parameters["Influence"],
        "rewiring": parameters["RewiringRate"],
        "repost_rate": parameters["RepostRate"],
        "tolerance": parameters["Tolerance"],
        "post_retain_count": metadata["PostRetainCount"],
        "complete": True,
        "stale": False,
    }


def _columns(endpoints):
    """Transpose actual scalar observation fields without changing their meanings."""
    keys = sorted({key for endpoint in endpoints for key in endpoint})
    result = {}
    for key in keys:
        values = [point.get(key) for point in endpoints]
        if values and all(isinstance(value, dict) for value in values):
            result[key] = _columns(values)
        elif all(value is None or isinstance(value, (bool, int, float)) for value in values):
            result[key] = np.array(
                [np.nan if value is None else value for value in values],
                dtype=np.float64 if any(value is None or isinstance(value, float) for value in values) else np.int64,
            )
        else:
            result[key] = values
    return result


def _flatten_arrays(values, prefix):
    arrays = {}
    for name, value in values.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif isinstance(value, dict):
            arrays.update(_flatten_arrays(value, key))
        else:
            raise ValueError(f"non-array observation field in real corpus: {key}")
    return arrays


def _actual_arm(arm, prefix):
    return {
        **_flatten_arrays(_columns(arm["endpoints"]), prefix),
        f"{prefix}.final_opinions": np.asarray(arm["final_opinions"], dtype=np.float64),
    }


def _actual_document(index, sources):
    path = sources["actual_files"][index]
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != sources["provenance"]["actual_records"][index]["sha256"]:
        raise ValueError(f"actual corpus source changed after inventory: {path}")
    envelope = json.loads(gzip.decompress(content))
    result = envelope["result"]
    task_id = path.stem.removesuffix(".json")
    metadata = deepcopy(sources["actual_tasks"][task_id]["metadata"])
    analysis_id = "actual-staged-summary-transform-v1"
    identifier = _identity(["actual-staged", task_id, analysis_id])
    payload = _actual_arm(result["reference"], "reference")
    raw_paths = [result["reference"]["raw_record"]]
    for phase, stage in result.get("stages", {}).items():
        for arm, value in stage["arms"].items():
            payload.update(_actual_arm(value, f"stages.{phase}.{arm}"))
        raw_paths.extend(value["raw_record"] for value in stage["arms"].values())
    return {
        **_scalars(
            metadata,
            identifier=identifier,
            origin="actual-staged-summary",
            mode="staged",
            name=task_id,
            analysis_id=analysis_id,
        ),
        "metadata": metadata,
        "protocol": deepcopy(sources["actual_protocol"]),
        "source": {
            "kind": "actual_exported_summary",
            "record": deepcopy(sources["provenance"]["actual_records"][index]),
            "plan_sha256": sources["provenance"]["actual_plan"]["sha256"],
            "task_fingerprint": envelope["task_fingerprint"],
            "raw_records_not_loaded": raw_paths,
            "transform": "numeric observation columns and final opinions; terminal/RNG metadata stays in source export",
        },
        "payload": payload,
    }


def _curve(generator, points, *, positive=False):
    time = np.linspace(0, 1, points)
    start, end = generator.uniform(0.05, 0.9, 2)
    rate = generator.uniform(1.5, 7)
    curve = start + (end - start) * (1 - np.exp(-rate * time)) / (1 - np.exp(-rate))
    for frequency in (1, 2, 4):
        curve += generator.normal(0, 0.018) * np.sin(frequency * np.pi * time + generator.uniform(-np.pi, np.pi))
    curve += generator.normal(0, 0.001, points)
    return np.asarray(np.maximum(curve, 0) if positive else curve, dtype=np.float64)


def _synthetic_source(template, task_id, revision):
    source = deepcopy(template)
    source["directory"] = f"synthetic-benchmark://{task_id}/revision-{revision}"
    source["synthetic"] = True
    source["identity_method"] = "synthetic layout modeled on the measured raw-file manifest; files do not exist"
    source["fingerprint"] = _identity([CORPUS_REVISION, task_id, revision, "source"])
    for ordinal, file in enumerate(source.get("files", [])):
        if "sha256" in file:
            file["sha256"] = _identity([task_id, revision, ordinal])
    return source


def _synthetic_document(index, sources):
    revision, task_index = divmod(index, FULL_PAPER_COUNT)
    task = sources["tasks"][task_index]
    generator = np.random.default_rng(np.random.SeedSequence([CORPUS_SEED, task_index, revision]))
    length = int(generator.choice(sources["lengths"], p=sources["length_probabilities"]))
    horizon = int(generator.integers(1024, 20001))
    steps = np.unique(np.linspace(0, horizon, length).astype(np.int64))
    length = len(steps)
    metadata = deepcopy(task["metadata"])
    centers = generator.uniform(-0.95, 0.95, int(generator.integers(1, 6)))
    terminal = generator.choice(centers, sources["final_opinion_size"])
    terminal += generator.normal(0, 0.001, len(terminal))
    payload = {
        "x_indices": steps,
        **{
            name: _curve(generator, length, positive=True)
            for name in (
                "p_index",
                "g_index",
                "h_index",
                "rho_edge",
                "rho_post",
                "legacy_p_index",
                "legacy_g_index",
                "legacy_h_index",
                "corrected_post_h_index",
                "mean_vars_smpl",
            )
        },
        "x_mean_vars": steps.copy(),
        "opinion_diff_seg_mean": generator.uniform(0.01, 0.3, 5).astype(np.float64),
        "opinion_diff_seg_std": generator.uniform(0.01, 0.1, 5).astype(np.float64),
        "final_opinions": np.sort(np.clip(terminal, -1, 1)).astype(np.float64),
    }
    if task["mode"] == "mech":
        probe = {}
        # Source packs sort JSON keys; consume the RNG in that same stable order.
        for name in sorted(sources["probe_shapes"]):
            shape = sources["probe_shapes"][name]
            if len(shape) == 1:
                probe[name] = np.linspace(-1 if name == "x" else 0, 1, shape[0], dtype=np.float64)
            else:
                t, x = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing="ij")
                probe[name] = np.asarray(
                    generator.normal() * x + generator.normal() * t + generator.normal(0, 0.005, shape),
                    dtype=np.float64,
                )
        payload.update({f"force_probe.{name}": value for name, value in probe.items()})
    analysis_id = f"synthetic-analysis-revision-{revision + 3}"
    origin = f"synthetic-paper-{task['mode']}"
    protocol = deepcopy(sources["schema"]["protocol"])
    protocol.update(benchmark_synthetic=True, corpus_seed=CORPUS_SEED, analysis_revision=revision + 3)
    protocol["verification"] = "synthetic storage corpus; no simulation or scientific validation"
    return {
        **_scalars(
            metadata,
            identifier=_identity([origin, task["id"], analysis_id]),
            origin=origin,
            mode=task["mode"],
            name=task["id"],
            analysis_id=analysis_id,
        ),
        "metadata": metadata,
        "protocol": protocol,
        "source": _synthetic_source(sources["schema"]["source"], task["id"], revision),
        "payload": payload,
    }


def document_at(index, profile="full_paper"):
    """Regenerate one document independently of iteration/worker scheduling."""
    if profile not in PROFILES:
        raise ValueError(f"unknown corpus profile: {profile}")
    sources = load_sources()
    count = (
        len(sources["actual_files"])
        if profile == "actual_available"
        else FULL_PAPER_COUNT * (2 if profile == "versioned_2x" else 1)
    )
    if type(index) is not int or not 0 <= index < count:
        raise IndexError(f"document index must be in [0,{count})")
    return _actual_document(index, sources) if profile == "actual_available" else _synthetic_document(index, sources)


def iter_documents(profile="full_paper", limit=None):
    count = get_profile(profile)["count"]
    if limit is not None:
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        count = min(count, limit)
    for index in range(count):
        yield document_at(index, profile)
