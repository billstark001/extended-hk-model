"""Collect existing opinion indices from saved raw histories without running SMP."""

import inspect
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from common.experiments.lifecycle import interruptible, session_lock
from common.experiments.plans import freeze_manifest
from common.io.json import atomic_json, fingerprint, read_result, save_result, sha
from common.runtime.process import run_tasks

from .core import IW_SAMPLING_METHOD

PATH_KEYS = ("profile", "task_id", "stage", "arm")
SOURCE_KEYS = ("plan_fingerprint", "binary_sha256", "raw_record")
POINT_COLUMNS = (*PATH_KEYS, "step", *SOURCE_KEYS, "I_p", "I_s", "I_h")
PATH_COLUMNS = (*PATH_KEYS, *SOURCE_KEYS, "I_w", "sampling_method", "prefix_start", "trajectory_end", "endpoint_complement")


def _source_configuration():
    from smp_bindings import RawSimulationRecord

    from common.numerics import adaptive_discrete_sampling, area_under_curve
    from common.numerics.kde import get_kde_pdf
    from ehk.metrics.density import FittedDensity
    from ehk.metrics.homophily import mean_followee_edge_concordance
    from ehk.metrics.polarization import DistanceCalculator
    from ehk.metrics.state import MeasurementState
    from ehk.metrics.trajectory import sample_indices
    from ehk.micro import observables

    from .evaluation import ExactMetricCache

    sources = {
        Path(inspect.getfile(item)).resolve()
        for item in (RawSimulationRecord, mean_followee_edge_concordance, DistanceCalculator, observables.calc_distance,
                     adaptive_discrete_sampling, area_under_curve, get_kde_pdf, ExactMetricCache,
                     MeasurementState, sample_indices, FittedDensity)
    }
    calculator = observables.distance_collector
    return {
        "collector_sha256": sha(Path(__file__)),
        "source_sha256": {str(path): sha(path) for path in sorted(sources)},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
        "distance_parameters": {
            name: getattr(calculator, name)
            for name in ("min_bandwidth", "sample_count", "max_kde_samples", "t_err", "k", "use_js_divergence")
        },
        "path_sampling": IW_SAMPLING_METHOD,
        "density_evaluation": "exact Gaussian KDE retained for the original metric comparison",
        "opinion_point_sampling": "exact_prepared_steps; existing_calc_distance_random_state_equals_step",
        "homophily_weighting": "positive_out_degree_users_equal_weight; existing_normalize_homophily",
        "prefix": "full_saved_history_from_step_zero",
    }


def _graph_snapshots(directory):
    """Recognize the same padded or unpadded snapshot names as RawSimulationRecord."""
    pattern = re.compile(r"graph-(\d+)\.msgpack(?:\.lz4)?$")
    return [
        (int(match.group(1)), path)
        for path in sorted(directory.glob("graph-*"))
        if path.is_file() and (match := pattern.fullmatch(path.name))
    ]


def _required_files(directory):
    required = ("metadata.json", "trajectory.json", "events.db", "observations.json")
    missing = [name for name in required if not (directory / name).is_file()]
    if not any(step == 0 for step, _ in _graph_snapshots(directory)):
        missing.append("graph-0.msgpack[.lz4]")
    if not any(path.is_file() for path in directory.glob("finished*")):
        missing.append("finished*")
    return missing


def _validate_raw(directory, path, points):
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("UniqueName") != directory.name:
        raise ValueError("raw metadata name differs from its directory")
    if float(metadata["HKParams"]["Tolerance"]) != float(path["epsilon"]):
        raise ValueError("raw confidence radius differs from the prepared task")
    if path.get("metadata_json"):
        expected = json.loads(path["metadata_json"])
        if metadata["HKParams"] != expected["HKParams"] or metadata["NodeCount"] != expected["NodeCount"]:
            raise ValueError("raw population or dynamics differ from the prepared task")
        if metadata.get("RNG") != expected.get("RNG"):
            raise ValueError("raw initialization seed differs from the prepared task")
        for field in ("PostRetainCount", "RecsysCount", "RecsysFactoryType"):
            if field in expected and metadata.get(field) != expected[field]:
                raise ValueError(f"raw {field} differs from the prepared task")
        parameters = path.get("expected_rec_params_json")
        expected_parameters = json.loads(parameters) if parameters else expected.get("RecSysParams")
        if expected_parameters is not None and metadata.get("RecSysParams") != expected_parameters:
            raise ValueError("raw recommendation parameters differ from the prepared arm")
    dataset = directory
    for _ in Path(path["raw_record"]).parts:
        dataset = dataset.parent
    binding_file = dataset / "runtime" / "binding.json"
    if binding_file.is_file():
        binding = json.loads(binding_file.read_text())
        actual_binary = (binding["arm_binary_sha256"].get(path["arm"])
                         if "arm_binary_sha256" in binding else binding.get("binary_sha256"))
        if actual_binary != path["binary_sha256"]:
            raise ValueError("mounted raw dataset binary differs from the prepared arm")
    observed = json.loads((directory / "observations.json").read_text())
    if int(observed["stopped_at"]) != int(path["end_step"]):
        raise ValueError("raw observation horizon differs from the prepared arm")
    recorded = {int(point["step"]): point["pathway"] for point in observed["endpoints"]}
    for point in points.to_dict("records"):
        state = recorded.get(int(point["step"]))
        if state is None:
            raise ValueError("prepared point is missing from the raw observations")
        for name in ("separation", "excess_concordance", "edge_concordance", "pair_concordance"):
            expected, actual = point[name], state[name]
            if pd.isna(expected) and actual is None:
                continue
            if actual is None or float(expected) != float(actual):
                raise ValueError(f"raw {name} differs from the authenticated prepared observation")
    return metadata


def _raw_sources(directory):
    # The trajectory manifest binds every compressed chunk by its own SHA;
    # TrajectoryReader verifies chunk contents when the existing indices read them.
    files = {directory / name for name in ("metadata.json", "observations.json", "trajectory.json", "events.db")}
    files.update(path for _, path in _graph_snapshots(directory))
    # Read-only SQLite connections still change SHM synchronization bytes.
    # The durable database and outstanding WAL carry the event contents.
    wal = directory / "events.db-wal"
    if wal.is_file():
        files.add(wal)
    return {path.name: sha(path) for path in sorted(files)}


def _measure(directory, path, points, metrics=None, *, endpoint_complement=True):
    from smp_bindings import RawSimulationRecord

    from .evaluation import ExactMetricCache

    if metrics is None:
        metrics = ExactMetricCache(cross_records=False)
    metadata = _validate_raw(directory, path, points)
    with RawSimulationRecord(str(directory.parent), metadata) as record:
        if record.max_step != int(path["end_step"]):
            raise ValueError("raw trajectory has a different final step")
        # Exact point comparisons use the saved observation times, not interpolation.
        values = []
        for point in points.to_dict("records"):
            step = int(point["step"])
            ip, isub = metrics.distance(record, step)
            ih = metrics.homophily(record, step)
            if not np.isfinite([ip, isub, ih]).all():
                raise ValueError(f"existing metric is not finite at step {step}")
            values.append({
                **{name: path[name] for name in (*PATH_KEYS, *SOURCE_KEYS)},
                "step": step, "I_p": float(ip), "I_s": float(isub), "I_h": float(ih),
            })
        # I_w is a completed-trajectory quantity. Keep the existing adaptive
        # sampling and signed integral, including the common pre-intervention prefix.
        steps, h, p, _ = metrics.indices(record)
        if not len(steps) or int(steps[0]) != 0 or int(steps[-1]) != record.max_step:
            raise ValueError("existing trajectory-index calculation omitted the full prefix")
        from ehk.micro.observables import get_gradation_index_hp

        iw = float(get_gradation_index_hp(p, h, endpoint_complement=endpoint_complement))
        if not np.isfinite(iw):
            raise ValueError("existing pathway index is not finite")
        trajectory = {
            **{name: path[name] for name in (*PATH_KEYS, *SOURCE_KEYS)},
            "I_w": iw, "sampling_method": IW_SAMPLING_METHOD,
            "prefix_start": 0, "trajectory_end": record.max_step,
            "endpoint_complement": endpoint_complement,
        }
    return {"points": values, "path": trajectory,
            "evaluation": {"adaptive_steps": [int(step) for step in steps], "endpoint_complement": endpoint_complement}}


def _collect_group(job):
    """One worker handles one initial state and shares only exact scalar evaluations."""
    from .evaluation import ExactMetricCache

    cache = ExactMetricCache()
    statuses = []
    output = Path(job["output"])
    for entry in job["paths"]:
        path, observations = entry["path"], entry["points"]
        identity = {name: path[name] for name in (*PATH_KEYS, *SOURCE_KEYS)}
        receipt_path = output / "records" / (fingerprint(identity) + ".json.gz")
        directory = Path(path["raw_directory"])
        if path["profile"] in job["raw_roots"]:
            directory = Path(job["raw_roots"][path["profile"]]) / path["raw_record"]
        missing = _required_files(directory)
        status = {**identity, "raw_directory": str(directory)}
        if receipt_path.is_file():
            result = read_result(receipt_path)
            if result["configuration_fingerprint"] != job["configuration_fingerprint"] or result["identity"] != identity:
                raise ValueError("cached legacy metrics differ from the preparation or implementation")
            if not missing and _raw_sources(directory) != result["raw_sha256"]:
                raise ValueError("raw source files changed after cached legacy metrics were calculated")
            status["status"] = "cached"
        elif missing:
            status.update(status="pending", missing_files=missing)
        else:
            try:
                sources = _raw_sources(directory)
                measured = _measure(directory, path, pd.DataFrame(observations), cache,
                                    endpoint_complement=job["endpoint_complement"])
                if _raw_sources(directory) != sources:
                    raise ValueError("raw source files changed during legacy metric calculation")
            except (OSError, ValueError, KeyError) as error:
                status.update(status="failed", error=str(error))
            else:
                save_result(receipt_path, {
                    "configuration_fingerprint": job["configuration_fingerprint"], "identity": identity,
                    "endpoint_complement": job["endpoint_complement"],
                    "raw_sha256": sources, "result": measured,
                })
                status["status"] = "complete"
        statuses.append(status)
    return {"paths": statuses, "cache_hits": cache.hits, "cache_misses": cache.misses}


def collect_raw(preparation, output, *, max_paths=None, raw_roots=None, workers=1, endpoint_complement=True):
    """Collect exact existing indices, with process workers and per-path recovery."""
    from .core import authenticate_preparation, endpoint_mode
    from ehk.metrics.protocol import metric_protocol

    preparation, output = Path(preparation).resolve(), Path(output).resolve()
    manifest = authenticate_preparation(preparation)
    endpoint_mode({"endpoint_complement": endpoint_complement})
    if max_paths is not None and (type(max_paths) is not int or max_paths < 1):
        raise ValueError("max_paths must be a positive integer")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer between 1 and 8")
    if output == preparation or preparation in output.parents or output in preparation.parents:
        raise ValueError("raw collection must use a separate output directory")
    raw_roots = {profile: str(Path(root).expanduser().resolve()) for profile, root in (raw_roots or {}).items()}
    paths = pd.read_csv(preparation / "paths.csv", keep_default_na=False, float_precision="round_trip")
    points = pd.read_csv(preparation / "points.csv.gz", keep_default_na=False, float_precision="round_trip", low_memory=False)
    groups = points.groupby(list(PATH_KEYS), sort=False).indices
    protocol = metric_protocol(endpoint_complement=endpoint_complement)
    protocol["distance_kde"].update(evaluation="exact", bins=None)
    configuration = {
        "schema": 1, "preparation_fingerprint": fingerprint(manifest),
        "metrics": _source_configuration(), "raw_roots": raw_roots,
        "endpoint_complement": endpoint_complement, "metric_protocol": protocol,
    }
    selected = paths if max_paths is None else paths.iloc[:max_paths]
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    outcomes, cache_counts = {}, {"hits": 0, "misses": 0}
    failure = None

    def jobs():
        for (profile, task), block in selected.groupby(["profile", "task_id"], sort=False):
            yield {
                "id": (profile, task), "output": str(output), "raw_roots": raw_roots,
                "configuration_fingerprint": fingerprint(configuration),
                "endpoint_complement": endpoint_complement,
                "paths": [{"path": row, "points": points.iloc[groups[tuple(row[name] for name in PATH_KEYS)]].to_dict("records")}
                          for row in block.to_dict("records")],
            }

    def progress(state, last=None):
        counts = {kind: sum(item["status"] == kind for item in outcomes.values())
                  for kind in ("complete", "cached", "pending", "failed")}
        atomic_json(output / "progress.json", {
            "state": state, "workers": workers, "planned_paths": len(paths), "selected_paths": len(selected),
            "completed_paths": counts["complete"] + counts["cached"], "counts": counts,
            "elapsed_seconds": time.monotonic() - started, "cache": cache_counts, "last_task": last,
        })

    def finished(job, result):
        for path in result["paths"]:
            outcomes[tuple(path[name] for name in PATH_KEYS)] = path
        cache_counts["hits"] += result["cache_hits"]
        cache_counts["misses"] += result["cache_misses"]
        progress("running", job["id"])

    with session_lock(output / "runtime"), interruptible():
        freeze_manifest(output / "configuration.json", configuration)
        progress("running")
        try:
            if workers == 1:
                for job in jobs():
                    finished(job, _collect_group(job))
            else:
                run_tasks(_collect_group, jobs(), workers, executor="process", on_result=finished)
        except BaseException as error:
            failure = error
        # Assemble committed path receipts even if an interrupt ended a worker.
        all_points, all_paths, statuses = [], [], []
        for path in selected.to_dict("records"):
            key = tuple(path[name] for name in PATH_KEYS)
            identity = {name: path[name] for name in (*PATH_KEYS, *SOURCE_KEYS)}
            receipt = output / "records" / (fingerprint(identity) + ".json.gz")
            status = outcomes.get(key, {**identity, "status": "pending"})
            if receipt.is_file():
                saved = read_result(receipt)
                if saved["configuration_fingerprint"] != fingerprint(configuration) or saved["identity"] != identity:
                    raise ValueError("path receipt disagrees with the frozen collection configuration")
                all_points.extend(saved["result"]["points"])
                all_paths.append(saved["result"]["path"])
                if status["status"] == "pending":
                    status = {**identity, "status": "complete"}
            statuses.append(status)
        pd.DataFrame(all_points, columns=POINT_COLUMNS).to_csv(output / "legacy_points.csv.gz", index=False)
        pd.DataFrame(all_paths, columns=PATH_COLUMNS).to_csv(output / "legacy_paths.csv", index=False)
        completed = len(all_paths)
        failed = sum(item["status"] == "failed" for item in statuses)
        report = {
            "schema": 1, "preparation_fingerprint": configuration["preparation_fingerprint"],
            "endpoint_complement": endpoint_complement,
            "preparation_sha256": sha(preparation / "preparation.json"),
            "status": ("interrupted" if isinstance(failure, KeyboardInterrupt) else "failed") if failure else
                      ("complete" if completed == len(paths) else ("failed" if failed else "pending")),
            "planned_paths": len(paths), "selected_paths": len(selected), "completed_paths": completed,
            "pending_paths": len(paths) - completed - failed, "failed_paths": failed,
            "points": len(all_points), "paths": statuses, "cache": cache_counts,
            "files": {name: sha(output / name) for name in ("legacy_points.csv.gz", "legacy_paths.csv", "configuration.json")},
        }
        atomic_json(output / "collection.json", report)
        progress(report["status"])
        if failure:
            raise failure
        return report
